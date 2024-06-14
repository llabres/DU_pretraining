from typing import Optional, Tuple, Union, List
import warnings

import copy

import numpy as np

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.distributed as dist

from transformers import T5TokenizerFast, T5ForConditionalGeneration, AutoFeatureExtractor, Pix2StructProcessor
from transformers.models.t5.modeling_t5 import T5Stack, T5Block
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateNonBeamOutput, GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)

from transformers.utils import logging
logger = logging.get_logger(__name__)


from .mp_vt5_collator import mp_vt5_collator, mp_vt5_collator_denoising
from models.modules import SpatialEmbeddings, ContinuousSpatialEmbeddings, VisualEmbeddings, sinusoidal_positional_embedding

class embeddings(nn.Module):
    def __init__(self, shared, spatial_embedding, visual_embedding, config):
        super().__init__()
        self.config = config
        self.shared = shared
        self.spatial_embedding = spatial_embedding
        self.visual_embedding = visual_embedding
        
    
    def forward(self, input_ids, boxes, images):
        input_ids = input_ids.view(-1, input_ids.size(-1))
        boxes = boxes.view(-1, boxes.size(-2), boxes.size(-1))
        images = images.view(-1, images.size(2), images.size(3))
        
        semantic_embedding = self.shared(input_ids)
        spatial_embedding = self.spatial_embedding(boxes)
        
        visual_embedding, visual_boxes = self.visual_embedding(images)
        visual_spatial_embedding = self.spatial_embedding(visual_boxes)
        visual_embeds = torch.add(visual_embedding, visual_spatial_embedding)
        
        inputs_embeds = torch.add(semantic_embedding, spatial_embedding)
        inputs_embeds = inputs_embeds.view(-1, self.config.max_pages, inputs_embeds.size(-2), inputs_embeds.size(-1))
        
        page_position = sinusoidal_positional_embedding(self.config.max_pages, self.config.hidden_size).to(inputs_embeds.device)
        page_position = page_position.unsqueeze(0).repeat(inputs_embeds.size(0), 1, 1).unsqueeze(2).repeat(1, 1, self.config.n_page_tokens, 1)
        inputs_embeds[:, :, -self.config.n_page_tokens:, :] = torch.add(inputs_embeds[:, :, -self.config.n_page_tokens:, :], page_position)
        inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))

        inputs_embeds = torch.cat([inputs_embeds[:, :-self.config.n_page_tokens, :], visual_embeds, inputs_embeds[:, -self.config.n_page_tokens:, :]], dim=1)

        return inputs_embeds

class MP_VT5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias):
        super().__init__()
        self.page_layer = T5Block(config, has_relative_attention_bias=has_relative_attention_bias)
        
        document_layer_config = copy.deepcopy(config)
        document_layer_config.d_ff = config.global_d_ff
        document_layer_config.num_heads = config.global_num_heads
        document_layer_config.d_kv = config.global_d_kv
        self.document_layer = T5Block(document_layer_config, has_relative_attention_bias=False)
        self.n_page_tokens = config.n_page_tokens
        self.max_pages = config.max_pages
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        
        page_output = self.page_layer(hidden_states, attention_mask, position_bias, encoder_hidden_states, encoder_attention_mask, encoder_decoder_position_bias, layer_head_mask, cross_attn_layer_head_mask, past_key_value, use_cache, output_attentions, return_dict)
        hidden_states = page_output[0]
        hidden_states = hidden_states.view(-1, self.max_pages, hidden_states.size(-2), hidden_states.size(-1))
        attention_mask = attention_mask.view(-1, self.max_pages, hidden_states.size(-2))


        page_tokens = torch.clone(hidden_states[:, :, -self.n_page_tokens:, :]).view(hidden_states.size(0), -1, hidden_states.size(-1))
        page_attention_mask = torch.clone(attention_mask[:, :, -self.n_page_tokens:]).view(hidden_states.size(0), 1, 1, -1)

        document_output = self.document_layer(page_tokens, page_attention_mask, output_attentions=output_attentions, return_dict=return_dict)

        hidden_states[:, :, -self.n_page_tokens:, :] = document_output[0].view(-1, self.max_pages, self.n_page_tokens, hidden_states.shape[-1])

        hidden_states = hidden_states.view(-1, hidden_states.size(-2), hidden_states.size(-1))

        return (hidden_states,) + page_output[1:]
    

class MP_VT5Stack(T5Stack):
    def __init__(self, config, embedding):
        super().__init__(config, embedding)

        self.block = nn.ModuleList(
            [MP_VT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens.shared = new_embeddings

    def forward(
        self,
        input_ids=None,
        boxes=None,
        images=None,
        page_idx_mask=None,
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

class MP_VT5(T5ForConditionalGeneration):
    _tied_weights_keys = ["shared.weight", "encoder.embed_tokens.shared.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)

        self.tokenizer = T5TokenizerFast.from_pretrained(config._name_or_path)
        self.AutoFeatureExtractor = Pix2StructProcessor.from_pretrained(config.feature_extractor_name)
        
        if config.pretraining:
            self.collator = mp_vt5_collator_denoising(self.tokenizer, self.AutoFeatureExtractor, config, padding=config.padding)
        else:
            self.collator = mp_vt5_collator(self.tokenizer, self.AutoFeatureExtractor, config, padding=config.padding)

        self.n_page_tokens = config.n_page_tokens
        self.max_pages = config.max_pages
        self.use_all_tokens = config.use_all_tokens

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
        embedding = embeddings(self.shared, spatial_embedding, visual_embedding, encoder_config)
        self.encoder = MP_VT5Stack(encoder_config, embedding)
            
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        boxes: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        page_idx_mask: Optional[torch.BoolTensor] = None,
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
        answer_page: Optional[torch.LongTensor] = None,
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
                page_idx_mask=page_idx_mask,
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

        if not self.use_all_tokens:
            page_tokens = hidden_states.view(-1, self.max_pages, hidden_states.size(-2), hidden_states.size(-1))[:, :, -self.n_page_tokens:, :]
            page_tokens = page_tokens.reshape(-1, self.max_pages*self.n_page_tokens, hidden_states.shape[-1])
            page_attention_mask = attention_mask.view(-1, self.max_pages, hidden_states.size(-2))[:, :, -self.n_page_tokens:]
            page_attention_mask = page_attention_mask.reshape(-1, self.max_pages*self.n_page_tokens)

            hidden_states = page_tokens
            attention_mask = page_attention_mask
        
        else:
            hidden_states = hidden_states.view(-1, self.max_pages*hidden_states.size(-2), hidden_states.size(-1))
            attention_mask = attention_mask.view(-1, self.max_pages*attention_mask.size(-1))


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

        if self.page_prediction is not None:
            answer_page_prediction = self.page_prediction(sequence_output)
        else:
            answer_page_prediction = None
        
        lm_logits = self.lm_head(sequence_output)

        loss = None
        logit_loss = None
        if labels is not None:
            # logit_loss = torch.tensor(0).to(lm_logits.device)
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            logit_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            loss = logit_loss
        
        answer_page_loss = None
        if answer_page is not None:
            #loss_fct = MSELoss()
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            answer_page = answer_page.to(answer_page_prediction.device)
            #answer_page_prediction = torch.clamp(answer_page_prediction[labels==self.tokenizer.get_vocab()['<page_prediction>']], min=-1e2, max=1e2)

            answer_page_prediction = torch.softmax(answer_page_prediction[labels==self.tokenizer.get_vocab()['<page_prediction>']], dim=-1)
            answer_page_loss = loss_fct(answer_page_prediction, answer_page)
            
            loss += answer_page_loss


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

        outputs['answer_page_prediction'] = answer_page_prediction
        outputs['losses'] = {
            'logit_loss': logit_loss,
            'answer_page_loss': answer_page_loss
        }

        return outputs

    def _greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin._greedy_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>


        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.

            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors
                for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
        >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> outputs = model._greedy_search(
        ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        if eos_token_id is not None:
            logger.warning_once(
                "`eos_token_id` is deprecated in this function and will be removed in v4.41, use"
                " `stopping_criteria=StoppingCriteriaList([EosTokenCriteria(eos_token_id=eos_token_id)])` instead."
                " Otherwise make sure to set `model.generation_config.eos_token_id`",
                FutureWarning,
            )
            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
        else:
            # TODO remove when the method is totally private
            # need to get `eos_token_id` and add stopping criteria, so that generation does not go forever
            eos_token_id = [
                criteria.eos_token_id.tolist() for criteria in stopping_criteria if hasattr(criteria, "eos_token_id")
            ]
            eos_token_id = eos_token_id[0] if eos_token_id else None
            if eos_token_id is None and self.generation_config.eos_token_id is not None:
                eos_token_id = self.generation_config.eos_token_id
                stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
        answer_page_prediction = None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            page_prediction = outputs['answer_page_prediction']

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )
                answer_page_prediction = page_prediction if answer_page_prediction is None else torch.cat([answer_page_prediction, page_prediction], dim=1)

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                output =  GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
                output['answer_page_prediction'] = answer_page_prediction
                return output
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
