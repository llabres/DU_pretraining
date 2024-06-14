import torch
import random

def process_patches(patches, max_patches):
    patches = patches.squeeze(0) # Remove he batch dimension

    image_width = patches[:, 1].max().item()
    image_height = patches[:, 0].max().item()

    for k, patch in enumerate(patches):
        if torch.std(patch[2:]) < 0.1:
            patches[k] = torch.zeros_like(patch)
    
    patches = patches[patches[:, 0] != 0]

    if patches.shape[0] > max_patches:
        patches = patches[torch.randperm(patches.shape[0])[:max_patches]]
    
    patches = torch.cat([torch.tensor([[image_width, image_height]]).repeat(patches.size(0), 1), patches], dim=1)
    return patches


class mp_vt5_collator:
    def __init__(self, tokenizer, image_processor, config, padding='longest'):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = self.tokenizer.model_max_length
        self.config = config
        self.padding = padding
        self.max_pages = config.max_pages
        self.n_page_tokens = config.n_page_tokens
        self.max_patches = config.max_patches
        self.image_resolution = config.image_resolution

    def __call__(self, batch):
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        batch_input_ids = []
        batch_input_boxes = []

        prefix_ids = torch.tensor(self.tokenizer.encode('question: ', add_special_tokens=False))
        suffix_ids = torch.tensor(self.tokenizer.encode('  context: ', add_special_tokens=False))
        for batch_idx in range(len(batch['question'])):
            pages_input_ids = []
            pages_input_boxes = []
            for page in range(len(batch['ocr_tokens'][batch_idx])):
                input_ids = torch.cat([prefix_ids, torch.tensor(self.tokenizer.encode(batch['question'][batch_idx].lower(), add_special_tokens=False)), suffix_ids])
                if self.config.continuous_spatial_embeddings:
                    input_boxes = torch.tensor([0, 0, 1, 1], dtype=torch.float32).repeat(len(input_ids), 1)
                else:
                    input_boxes = torch.tensor([0, 0, 1000, 1000], dtype=torch.long).repeat(len(input_ids), 1)
                for word, box in zip(batch['ocr_tokens'][batch_idx][page], batch['ocr_boxes'][batch_idx][page]):
                    word = word.lower()
                    word_ids = torch.tensor(self.tokenizer.encode(word, add_special_tokens=False))
                    input_ids = torch.cat([input_ids, word_ids])
                    if self.config.continuous_spatial_embeddings:
                        input_boxes = torch.cat([input_boxes, torch.tensor(box).repeat(len(word_ids), 1)])
                    else:
                        input_boxes = torch.cat([input_boxes, (torch.tensor(box)*1000).to(torch.long).repeat(len(word_ids), 1)])
                input_ids = input_ids[:self.max_length-1-self.n_page_tokens]
                input_boxes = input_boxes[:self.max_length-1-self.n_page_tokens]
                
                # Add the eos token
                input_ids = torch.cat([input_ids, torch.tensor([self.tokenizer.eos_token_id])])
                input_boxes = torch.cat([input_boxes, torch.tensor([[0, 0, 0, 0]], dtype=torch.long)])

                pages_input_ids.append(input_ids)
                pages_input_boxes.append(input_boxes)

            batch_input_ids.append(pages_input_ids)
            batch_input_boxes.append(pages_input_boxes)

        # Add padding
        if self.padding == 'longest':
            longest = max([len(page) + self.n_page_tokens for document in batch_input_ids for page in document])
            max_length = longest if longest < self.max_length else self.max_length
        else:
            max_length = self.max_length
        
        documents_input_ids = []
        documents_input_boxes = []
        documents_visual_patches = []

        page_idx_mask = [] # batch_size*max_pages
        for document_ids, document_boxes, document_images in zip(batch_input_ids, batch_input_boxes, batch['images']):
            pages_input_ids = []
            pages_input_boxes = []
            pages_image_patches = []
            for page_ids, page_boxes, page_image in zip(document_ids, document_boxes, document_images):
                page_tokens = "".join([f'<page_token_{i}>' for i in range(self.n_page_tokens)])
                pages_input_ids.append(torch.cat([page_ids, torch.tensor([self.tokenizer.pad_token_id]).repeat(max_length-len(page_ids)-self.n_page_tokens), torch.tensor(self.tokenizer.encode(page_tokens, add_special_tokens=False))]))
                if self.config.continuous_spatial_embeddings:
                    pages_input_boxes.append(torch.cat([page_boxes, torch.tensor([[0, 0, 0, 0]], dtype=torch.long).repeat(max_length-len(page_boxes)-self.n_page_tokens, 1), torch.tensor([[0, 0, 1, 1]]*self.n_page_tokens)]))
                else:
                    pages_input_boxes.append(torch.cat([page_boxes, torch.tensor([[0, 0, 0, 0]], dtype=torch.long).repeat(max_length-len(page_boxes)-self.n_page_tokens, 1), torch.tensor([[0, 0, 1000, 1000]]*self.n_page_tokens)]))

                patches = self.image_processor(page_image, return_tensors='pt', max_patches=self.image_resolution)['flattened_patches']
                pages_image_patches.append(process_patches(patches, self.max_patches))

            page_idx_mask.extend([True]*len(pages_input_ids))
            if len(pages_input_ids) < self.max_pages:
                page_idx_mask.extend([False]*(self.max_pages-len(pages_input_ids)))
                pages_input_ids += [torch.tensor([self.tokenizer.pad_token_id]*max_length)]*(self.max_pages-len(pages_input_ids))
                pages_input_boxes += [torch.tensor([[0, 0, 0, 0]]*max_length)]*(self.max_pages-len(pages_input_boxes))
                pages_image_patches += [torch.zeros_like(pages_image_patches[0])]*(self.max_pages-len(pages_image_patches))
                

            documents_input_ids.append(torch.stack(pages_input_ids))
            documents_input_boxes.append(torch.stack(pages_input_boxes))
            documents_visual_patches.append(pages_image_patches)
        
        # Add padding to the image patches
        if self.padding == 'longest':
            max_patches = max([len(patches) for document_patches in documents_visual_patches for patches in document_patches])
        else:
            max_patches = self.max_patches
        
        images = torch.stack([
                    torch.stack([
                        torch.cat([ patches, torch.zeros((max_patches - patches.size(0), patches.size(1)))], dim=0) if patches.size(0) < max_patches else patches 
                        for patches in document_patches], dim=0) 
                    for document_patches in documents_visual_patches], dim=0)
        
        input_ids = torch.stack(documents_input_ids)
        input_boxes = torch.stack(documents_input_boxes)


        visual_attention_mask = (images[:, :, :, 2] != 0).to(torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(torch.long)
        attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=-1)
        
        labels = self.tokenizer(batch['label'], padding='longest', return_tensors='pt', add_special_tokens=True, truncation=True)
        decoder_attention_mask = labels.attention_mask[:, :50]
        labels = labels.input_ids[:, :50]
        # set padding token to -100 so they are not taken into account in the loss
        labels[labels == self.tokenizer.pad_token_id] = -100
    
        return dict(
            input_ids=input_ids.to(torch.long),
            attention_mask=attention_mask,
            labels=labels,
            answer_page = torch.tensor(batch['answer_page_idx'], dtype=torch.long) if self.config.page_prediction else None,
            boxes=input_boxes.to(torch.long) if not self.config.continuous_spatial_embeddings else input_boxes,
            decoder_attention_mask=decoder_attention_mask,
            images=images,
            page_idx_mask=torch.tensor(page_idx_mask),
            gt_answers=batch.get('gt_answers', None),
            gt_answer_page=batch.get('gt_answer_page', None),
        )

class mp_vt5_collator_denoising:
    def __init__(self, tokenizer, image_processor, config, padding='longest'):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = self.tokenizer.model_max_length
        self.config = config
        self.padding = padding
        self.max_pages = config.max_pages
        self.n_page_tokens = config.n_page_tokens
        self.max_patches = config.max_patches
        self.image_resolution = config.image_resolution

    def __call__(self, batch):
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        batch_size = len(batch['ocr_tokens'])

        batch_input_ids = []
        batch_input_boxes = []
        batch_labels = []
        batch_images = []
        for batch_idx in range(batch_size):
            pages_input_ids = []
            pages_input_boxes = []
            page_labels = []
            page_images = []
            num_pages = len(batch['ocr_tokens'][batch_idx])
            for page in range(num_pages):
                input_ids = torch.tensor([])
                input_boxes = torch.tensor([])
                for word, box in zip(batch['ocr_tokens'][batch_idx][page], batch['ocr_boxes'][batch_idx][page]):
                    word = word.lower()
                    word_ids = torch.tensor(self.tokenizer.encode(word, add_special_tokens=False))
                    input_ids = torch.cat([input_ids, word_ids])
                    if self.config.continuous_spatial_embeddings:
                        input_boxes = torch.cat([input_boxes, torch.tensor(box).repeat(len(word_ids), 1)])
                    else:
                        input_boxes = torch.cat([input_boxes, (torch.tensor(box)*1000).to(torch.long).repeat(len(word_ids), 1)])
                input_ids = input_ids[:self.max_length-1-self.n_page_tokens]
                input_boxes = input_boxes[:self.max_length-1-self.n_page_tokens]
                
                # Add the eos token
                input_ids = torch.cat([input_ids, torch.tensor([self.tokenizer.eos_token_id])])
                input_boxes = torch.cat([input_boxes, torch.tensor([[0, 0, 0, 0]], dtype=torch.long)])

                mask_spans = []
                i = 0
                while i < len(input_ids) - 1:  # last token is EOS, which we don't want to mask
                    # TODO: Set this to self.args.mlm_probability : 0.15, it is hardcoded right now.
                    if len(mask_spans) < 100 // num_pages and random.random() < 0.15 * 0.333:
                        start = i
                        end = i + random.randint(1, 5)  # create a span of 1， 2 or 3 or 4， 5.
                        end = min(end, len(input_ids) - 2)
                        mask_spans.append([start, end])
                        i = end + 1
                    else:
                        i += 1
                
                mask_ID_counter = 0
                new_input_ids = torch.tensor([])
                new_input_boxes = torch.tensor([])
                labels = torch.tensor([])
                previous_end = 0
                image = batch['image'][batch_idx][page]
                image_size = image.size

                for start, end in mask_spans:
                    extra_id = torch.tensor([self.tokenizer.convert_tokens_to_ids(f"<extra_id_{mask_ID_counter}>")])
                    labels = torch.cat([labels, extra_id, input_ids[start:end+1]])
                    new_input_ids = torch.cat([new_input_ids, input_ids[previous_end:start], extra_id])
                    new_input_boxes = torch.cat([new_input_boxes, input_boxes[previous_end:start],
                                                torch.tensor([[torch.min(input_boxes[start:end+1][:, 0]), torch.min(input_boxes[start:end+1][:, 1]),
                                                               torch.max(input_boxes[start:end+1][:, 2]), torch.max(input_boxes[start:end+1][:, 3])]])])
                    
            
                    
                    previous_end = end + 1
                    mask_ID_counter += 1
                
                new_input_ids = torch.cat([new_input_ids, input_ids[previous_end:]])
                new_input_boxes = torch.cat([new_input_boxes, input_boxes[previous_end:]])

                pages_input_ids.append(new_input_ids)
                pages_input_boxes.append(new_input_boxes)
                page_labels.append(labels)
                page_images.append(image)

            batch_input_ids.append(pages_input_ids)
            batch_input_boxes.append(pages_input_boxes)
            batch_labels.append(page_labels)
            batch_images.append(page_images)

        # Add padding
        if self.padding == 'longest':
            longest = max([len(page) + self.n_page_tokens for document in batch_input_ids for page in document])
            max_length = longest if longest < self.max_length else self.max_length
            label_longest = max([len(label) for document in batch_labels for label in document])
            max_label_length = label_longest if label_longest < int(self.max_length*0.25) else int(self.max_length*0.25)
        else:
            max_length = self.max_length
            max_label_length = int(self.max_length*0.25)
        
        documents_input_ids = []
        documents_input_boxes = []
        documents_visual_patches = []
        documents_labels = []
        
        for document_ids, document_boxes, document_labels, document_images in zip(batch_input_ids, batch_input_boxes, batch_labels, batch_images):
            pages_input_ids = []
            pages_input_boxes = []
            pages_labels = []
            page_visual_patches = []

            for page_ids, page_boxes, page_labels, image in zip(document_ids, document_boxes, document_labels, document_images):
                page_tokens = "".join([f'<page_token_{i}>' for i in range(self.n_page_tokens)])
                pages_input_ids.append(torch.cat([page_ids, torch.tensor([self.tokenizer.pad_token_id]).repeat(max_length-len(page_ids)-self.n_page_tokens), torch.tensor(self.tokenizer.encode(page_tokens, add_special_tokens=False))]))
                if self.config.continuous_spatial_embeddings:
                    pages_input_boxes.append(torch.cat([page_boxes, torch.tensor([[0, 0, 0, 0]], dtype=torch.long).repeat(max_length-len(page_boxes)-self.n_page_tokens, 1), torch.tensor([[0, 0, 1, 1]]*self.n_page_tokens)]))
                else:
                    pages_input_boxes.append(torch.cat([page_boxes, torch.tensor([[0, 0, 0, 0]], dtype=torch.long).repeat(max_length-len(page_boxes)-self.n_page_tokens, 1), torch.tensor([[0, 0, 1000, 1000]]*self.n_page_tokens)]))
                
                page_labels = page_labels[:max_label_length]
                pages_labels.append(torch.cat([page_labels, torch.tensor([self.tokenizer.pad_token_id]).repeat(max_label_length-len(page_labels))]))

                patches = self.image_processor(image, return_tensors='pt', max_patches=self.image_resolution)['flattened_patches']
                page_visual_patches.append(process_patches(patches, self.max_patches))

            if len(pages_input_ids) < self.max_pages:
                pages_input_ids += [torch.tensor([self.tokenizer.pad_token_id]*max_length)]*(self.max_pages-len(pages_input_ids))
                pages_input_boxes += [torch.tensor([[0, 0, 0, 0]]*max_length)]*(self.max_pages-len(pages_input_boxes))
                page_visual_patches += [torch.zeros_like(page_visual_patches[0])]*(self.max_pages-len(page_visual_patches))
                

            documents_input_ids.append(torch.stack(pages_input_ids))
            documents_input_boxes.append(torch.stack(pages_input_boxes))
            documents_labels.append(torch.stack(pages_labels))
            documents_visual_patches.append(page_visual_patches)
        
        if self.padding == 'longest':
            max_patches = max([len(patches) for document_patches in documents_visual_patches for patches in document_patches])

        else:
            max_patches = self.max_patches

        images = torch.stack([
                    torch.stack([
                        torch.cat([ patches, torch.zeros((max_patches - patches.size(0), patches.size(1)))], dim=0) if patches.size(0) < max_patches else patches 
                        for patches in document_patches], dim=0) 
                    for document_patches in documents_visual_patches], dim=0)


        input_ids = torch.stack(documents_input_ids)
        input_boxes = torch.stack(documents_input_boxes)
        
        labels = torch.stack(documents_labels).view(batch_size, -1)
        labels[labels == self.tokenizer.pad_token_id] = -100
        decoder_attention_mask = (labels != -100).to(torch.long)

        visual_attention_mask = (images[:, :, :, 0] != 0).to(torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(torch.long)
        attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=-1)

        return dict(
            input_ids=input_ids.to(torch.long),
            attention_mask=attention_mask,
            labels=labels.to(torch.long),
            boxes=input_boxes.to(torch.long) if not self.config.continuous_spatial_embeddings else input_boxes,
            decoder_attention_mask=decoder_attention_mask,
            images=images,
        )



        
    
