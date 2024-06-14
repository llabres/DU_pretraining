from typing import Optional, Tuple, Union, List
import warnings

import copy

import numpy as np

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.distributed as dist

from transformers import T5TokenizerFast, T5ForConditionalGeneration, AutoFeatureExtractor, Pix2StructProcessor
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput



from .vt5_collator import vt5_collator, vt5_collator_denoising
from models.modules import SpatialEmbeddings, ContinuousSpatialEmbeddings, VisualEmbeddings

class embeddings(nn.Module):
    def __init__(self, shared, spatial_embedding, visual_embedding):
        super().__init__()
        self.shared = shared
        self.spatial_embedding = spatial_embedding
        self.visual_embedding = visual_embedding
    
    def forward(self, input_ids, boxes, images):
        semantic_embedding = self.shared(input_ids)
        spatial_embedding = self.spatial_embedding(boxes)
        inputs_embeds = torch.add(semantic_embedding, spatial_embedding)
        
        visual_embedding, visual_boxes = self.visual_embedding(images)
        visual_spatial_embedding = self.spatial_embedding(visual_boxes)
        visual_embeds = torch.add(visual_embedding, visual_spatial_embedding)
        
        inputs_embeds = torch.cat([inputs_embeds, visual_embeds], dim=1)

        return inputs_embeds


class VT5Stack(T5Stack):
    def __init__(self, config, embedding):
        super().__init__(config, embedding)
    
    def forward(
        self,
        input_ids=None,
        boxes=None,
        images=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        inputs_embeds = self.embed_tokens(input_ids, boxes, images)

        outputs = super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        outputs['attention_mask'] = attention_mask

        return outputs

class VT5(T5ForConditionalGeneration):
    _tied_weights_keys = ["shared.weight", "encoder.embed_tokens.shared.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)

        self.tokenizer = T5TokenizerFast.from_pretrained(config._name_or_path)
        #self.AutoFeatureExtractor = AutoFeatureExtractor.from_pretrained(config.feature_extractor_name)
        self.AutoFeatureExtractor = Pix2StructProcessor.from_pretrained(config.feature_extractor_name)
        
        if config.pretraining:
            self.collator = vt5_collator_denoising(self.tokenizer, self.AutoFeatureExtractor, config, padding=config.padding)
        else:
            self.collator = vt5_collator(self.tokenizer, self.AutoFeatureExtractor, config, padding=config.padding)

        if config.continuous_spatial_embeddings:
            spatial_embedding = ContinuousSpatialEmbeddings(config)

        else:
            spatial_embedding = SpatialEmbeddings(config)

        if config.page_prediction:
            self.page_prediction = nn.Linear(config.hidden_size, 2)
        else:
            self.page_prediction = None
        
        config.patch_embed_hidden_size = 16*16*3
        visual_embedding = VisualEmbeddings(config)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        embedding = embeddings(self.shared, spatial_embedding, visual_embedding)
        self.encoder = VT5Stack(encoder_config, embedding)

            
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        boxes: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        target_logits: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                __HEAD_MASK_WARNING_MSG = """
                The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
                `decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
                If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
                num_heads)`.
                """
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                boxes=boxes,
                images=images,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            attention_mask = encoder_outputs['attention_mask']

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            attention_mask = encoder_outputs['attention_mask']
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        
        hidden_states = encoder_outputs[0]
        
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            if target_logits is not None:
                loss = 0
                target_logits = target_logits.to(lm_logits.device)
                lm_logits = torch.softmax(lm_logits, dim=-1)
                target_logits = torch.softmax(target_logits, dim=-1)

                for i in range(len(target_logits)):
                    loss_fct = CrossEntropyLoss()
                    loss += loss_fct(lm_logits[i][labels[i]!=-100].view(-1, lm_logits[i].size(-1)),
                                     target_logits[i][labels[i]!=-100].view(-1, lm_logits[i].size(-1)))
            else:
                loss = 0
                # logit_loss = torch.tensor(0).to(lm_logits.device)
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                # move labels to correct device to enable PP
                labels = labels.to(lm_logits.device)
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        outputs =  Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

        return outputs
