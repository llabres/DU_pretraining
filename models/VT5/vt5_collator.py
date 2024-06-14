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

class vt5_collator:
    def __init__(self, tokenizer, image_processor, config, padding='longest'):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = self.tokenizer.model_max_length
        self.config = config
        self.padding = padding
        self.max_patches = config.max_patches
        self.image_resolution = config.image_resolution

    def __call__(self, batch):
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        batch_input_ids = []
        batch_input_boxes = []
        batch_images = []
        prefix_ids = torch.tensor(self.tokenizer.encode('question: ', add_special_tokens=False))
        suffix_ids = torch.tensor(self.tokenizer.encode('  context: ', add_special_tokens=False))
        for batch_idx in range(len(batch['question'])):
            input_ids = torch.cat([prefix_ids, torch.tensor(self.tokenizer.encode(batch['question'][batch_idx].lower(), add_special_tokens=False)), suffix_ids])
            if self.config.continuous_spatial_embeddings:
                input_boxes = torch.tensor([0, 0, 1, 1], dtype=torch.float32).repeat(len(input_ids), 1)
            else:
                input_boxes = torch.tensor([0, 0, 1000, 1000], dtype=torch.long).repeat(len(input_ids), 1)
            for word, box in zip(batch['ocr_tokens'][batch_idx], batch['ocr_boxes'][batch_idx]):
                word = word.lower()
                word_ids = torch.tensor(self.tokenizer.encode(word, add_special_tokens=False))
                input_ids = torch.cat([input_ids, word_ids])
                if self.config.continuous_spatial_embeddings:
                    input_boxes = torch.cat([input_boxes, torch.tensor(box).repeat(len(word_ids), 1)])
                else:
                    input_boxes = torch.cat([input_boxes, (torch.tensor(box)*1000).to(torch.long).repeat(len(word_ids), 1)])
            input_ids = input_ids[:self.max_length-1]
            input_boxes = input_boxes[:self.max_length-1]
            
            # Add the eos token
            input_ids = torch.cat([input_ids, torch.tensor([self.tokenizer.eos_token_id])])
            input_boxes = torch.cat([input_boxes, torch.tensor([[0, 0, 0, 0]], dtype=torch.long)])

            patches = self.image_processor(batch['images'], return_tensors='pt', max_patches=self.image_resolution)['flattened_patches']
            batch_images.append(process_patches(patches, self.max_patches))

            batch_input_ids.append(input_ids)
            batch_input_boxes.append(input_boxes)
            batch_images.append(patches)

        # Add padding
        if self.padding == 'longest':
            longest = max([len(x) for x in batch_input_ids])
            max_length = longest if longest < self.max_length else self.max_length
            max_patches = max([len(image) for image in batch_images])

        else:
            max_length = self.max_length
            max_patches = self.max_patches

        input_ids = torch.stack([torch.cat([x, torch.tensor([self.tokenizer.pad_token_id]).repeat(max_length-len(x))]) for x in batch_input_ids])
        input_boxes = torch.stack([torch.cat([x, torch.tensor([[0, 0, 0, 0]], dtype=torch.long).repeat(max_length-len(x), 1)]) for x in batch_input_boxes])
        images = torch.stack([torch.cat([image, torch.zeros((max_patches - image.size(0), image.size(1)))], dim=0) if image.size(0) < max_patches else image for image in batch_images])

        visual_attention_mask = (images[:, :, :, 2] != 0).to(torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(torch.long)
        attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=-1)

        labels = self.tokenizer(batch['label'], padding='longest', return_tensors='pt', add_special_tokens=True, truncation=True, max_length=8)
        decoder_attention_mask = labels.attention_mask
        labels = labels.input_ids
        # set padding token to -100 so they are not taken into account in the loss
        labels[labels == self.tokenizer.pad_token_id] = -100

    
        if 'target_logits' in batch:
            target_logits = []
            for i in range(len(batch['target_logits'])):
                mask = (labels[i] == -100).nonzero().flatten()
                mask = mask[0] if len(mask) > 0 else len(labels[i])
                target_logits.append(torch.tensor(batch['target_logits'][i][:mask]))
            
            # add padding
            target_logits = torch.stack([torch.cat([x, torch.zeros(len(labels[0])-len(x), x.shape[1])]) for x in target_logits])


        return dict(
            input_ids=input_ids.to(torch.long),
            attention_mask=attention_mask,
            labels=labels,
            boxes=input_boxes.to(torch.long) if not self.config.continuous_spatial_embeddings else input_boxes,
            decoder_attention_mask=decoder_attention_mask,
            images=images,
            gt_answers=batch.get('gt_answers', None),
            target_logits=target_logits if 'target_logits' in batch else None
        )

class vt5_collator_denoising:
    def __init__(self, tokenizer, image_processor, config, padding='longest'):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = self.tokenizer.model_max_length
        self.config = config
        self.padding = padding
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
            input_ids = torch.tensor([])
            input_boxes = torch.tensor([])
            for word, box in zip(batch['ocr_tokens'][batch_idx], batch['ocr_boxes'][batch_idx]):
                word = word.lower()
                word_ids = torch.tensor(self.tokenizer.encode(word, add_special_tokens=False))
                input_ids = torch.cat([input_ids, word_ids])
                if self.config.continuous_spatial_embeddings:
                    input_boxes = torch.cat([input_boxes, torch.tensor(box).repeat(len(word_ids), 1)])
                else:
                    input_boxes = torch.cat([input_boxes, (torch.tensor(box)*1000).to(torch.long).repeat(len(word_ids), 1)])
            input_ids = input_ids[:self.max_length-1]
            input_boxes = input_boxes[:self.max_length-1]
            
            # Add the eos token
            input_ids = torch.cat([input_ids, torch.tensor([self.tokenizer.eos_token_id])])
            input_boxes = torch.cat([input_boxes, torch.tensor([[0, 0, 0, 0]], dtype=torch.long)])

            mask_spans = []
            i = 0
            while i < len(input_ids) - 1:  # last token is EOS, which we don't want to mask
                # TODO: Set this to self.args.mlm_probability : 0.15, it is hardcoded right now.
                if len(mask_spans) < 100 and random.random() < 0.15 * 0.333:
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

            batch_input_ids.append(new_input_ids)
            batch_input_boxes.append(new_input_boxes)
            batch_labels.append(labels)
            patches = self.image_processor(batch['image'][batch_idx], return_tensors='pt', max_patches=self.image_resolution)['flattened_patches']
            batch_images.append(process_patches(patches, self.max_patches))

        # Add padding
        if self.padding == 'longest':
            longest = max([len(x) for x in batch_input_ids])
            max_length = longest if longest < self.max_length else self.max_length
            label_longest = max([len(label) for label in batch_labels])
            max_label_length = label_longest if label_longest < int(self.max_length*0.25) else int(self.max_length*0.25)
            max_patches = max([len(image) for image in batch_images])
        else:
            max_length = self.max_length
            max_label_length = int(self.max_length*0.25)
            max_patches = self.max_patches
        
        documents_input_ids = []
        documents_input_boxes = []
        documents_labels = []
        documents_images = []
        for ids, boxes, labels, image in zip(batch_input_ids, batch_input_boxes, batch_labels, batch_images):
            documents_input_ids.append(torch.cat([ids, torch.tensor([self.tokenizer.pad_token_id]).repeat(max_length-len(ids))]))
            if self.config.continuous_spatial_embeddings:
                documents_input_boxes.append(torch.cat([boxes, torch.tensor([[0, 0, 0, 0]], dtype=torch.long).repeat(max_length-len(boxes), 1)]))
            else:
                documents_input_boxes.append(torch.cat([boxes, torch.tensor([[0, 0, 0, 0]], dtype=torch.long).repeat(max_length-len(boxes), 1)]))
            image = torch.cat([image, torch.zeros((max_patches - image.size(0), image.size(1)))], dim=0) if image.size(0) < max_patches else image
            documents_images.append(image)
            labels = labels[:max_label_length]
            documents_labels.append(torch.cat([labels, torch.tensor([self.tokenizer.pad_token_id]).repeat(max_label_length-len(labels))]))

        input_ids = torch.stack(documents_input_ids)
        input_boxes = torch.stack(documents_input_boxes)
        images = torch.stack(documents_images)

        labels = torch.stack(documents_labels)
        labels[labels == self.tokenizer.pad_token_id] = -100
        decoder_attention_mask = (labels != -100).to(torch.long)

        visual_attention_mask = (images[:, :, 2] != 0).to(torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(torch.long)
        attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)

        return dict(
            input_ids=input_ids.to(torch.long),
            attention_mask=attention_mask,
            labels=labels.to(torch.long),
            boxes=input_boxes.to(torch.long) if not self.config.continuous_spatial_embeddings else input_boxes,
            decoder_attention_mask=decoder_attention_mask,
            images=images,
        )