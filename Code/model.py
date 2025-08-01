'''
Adapted from https://github.com/huggingface/transformers
'''
from transformers import T5Config, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack, __HEAD_MASK_WARNING_MSG, T5EncoderModel
import copy
import math
import os
import warnings
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
)
from transformers.modeling_outputs import BaseModelOutput
from torch.nn.functional import cross_entropy
import torch.nn.functional as F

# 导入必要的库
from torch.nn.init import trunc_normal_

class ContextDecoder(nn.Module):
    def __init__(self, transformer_width=256, transformer_heads=16, transformer_layers=6, visual_dim=768, dropout=0.1, **kwargs):
        super().__init__()

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )

        # 使用 MultiheadAttention 替代自定义的 Attention
        self.mha_layer = torch.nn.MultiheadAttention(embed_dim=transformer_width, kdim=transformer_width, vdim=transformer_width, num_heads=transformer_heads, batch_first=True)

        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, text, visual):
        B, N, C = visual.shape
        visual = self.memory_proj(visual)
        x = self.text_proj(text)

        # 使用 MultiheadAttention 替代自定义的 Attention
        x, _ = self.mha_layer(x, visual, visual)

        return self.out_proj(x)


# 替代原来的 SemanticAligner 类
class SemanticAligner(nn.Module):
    def __init__(self, transformer_width=128, transformer_heads=16, ffw_width = 256, transformer_layers=1, visual_dim=768, dropout=0.2):
        super().__init__()
        self.context_decoder = ContextDecoder(
            transformer_width=transformer_width,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            visual_dim=visual_dim,
            dropout=dropout
        )

        # 定义一个 feedforward 网络层
        self.ffd_layer = nn.Sequential(
            nn.Linear(visual_dim, ffw_width),  # 第一个线性层
            nn.ReLU(),  # 激活函数
            nn.Dropout(dropout),  # Dropout 正则化
            nn.Linear(ffw_width, visual_dim)  # 第二个线性层
        )

        #self.ffd_layer = nn.Sequential(nn.Linear(transformer_width, ffw_width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ffw_width, transformer_width))
        self.gamma = nn.Parameter(torch.ones(visual_dim) * 1e-4)
        self.alpha_dense = nn.Parameter(torch.ones(visual_dim) * 1e-4)  # 注意维度要与 ffd_layer 输出匹配
        #self.alpha = nn.Parameter(torch.ones(visual_dim) * 1e-4)

    def forward(self, text_embeddings, visual_context):
        text_diff = self.context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + torch.tanh(self.gamma) * text_diff

        # 调用 feedforward 网络层，并使用 alpha_dense 进行调节
        ffd_output = self.ffd_layer(text_embeddings)
        text_embeddings = text_embeddings + torch.tanh(self.alpha_dense) * ffd_output
        #text_embeddings = text_embeddings + self.alpha * self.ffd_layer(text_embeddings)

        return text_embeddings


class T5ForMultimodalGeneration(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config, patch_size, padding_idx, save_dir,vot_num,alpha, freeze_unet, freeze_vae, freeze_encoder,
                 freeze_decoder):
        super().__init__(config)
        self.model_dim = config.d_model
        self.vot_num=vot_num
        self.alpha=alpha
        self.padding_idx = padding_idx

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.patch_num, self.patch_dim = patch_size

        self.image_dense = nn.Linear(self.patch_dim, config.d_model)
        self.mha_layer = torch.nn.MultiheadAttention(embed_dim=config.hidden_size, kdim=config.hidden_size, vdim=config.hidden_size, num_heads=1, batch_first=True)
        self.gate_dense = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.sigmoid = nn.Sigmoid()

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # 新增多模态融合组件
        self.semantic_aligner = SemanticAligner(
            transformer_width=128,  # 根据需要进行调整
            transformer_heads=16,  # 根据需要进行调整
            transformer_layers=1,  # 根据需要进行调整
            visual_dim=config.d_model,  # 与模型维度保持一致
            dropout=0.1  # 根据需要进行调整
        )

        if  freeze_vae:
            for param in self.vae.parameters():
                param.requires_grad = False

        if freeze_unet:
            for param in self.unet.parameters():
                param.requires_grad = False


        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False  # 冻结文本编码器的参数

        if freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_ids=None,
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
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        decoder_input_ids_base=decoder_input_ids
        # decoder_attention_mask_base=decoder_attention_mask
        decoder_inputs_embeds_base=decoder_inputs_embeds
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=True,  # 设置这里 output_attentions
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # hidden_states = encoder_outputs[0]
        hidden_states = encoder_outputs.last_hidden_state

        # 多模态融合
        image_embedding = self.image_dense(image_ids)
        # print(image_embedding.shape)
        hidden_states = self.semantic_aligner(hidden_states, image_embedding)
        # refined_textual_embedding = self.semantic_aligner(encoder_outputs.last_hidden_state, image_embedding)

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
        all_logits = []

        for _ in range(self.vot_num):
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
            all_logits.append(lm_logits)

        # voting        
        stacked_logits = torch.stack(all_logits, dim=0)
        mean_logits = torch.mean(stacked_logits, dim=0)
        stddev_logits = torch.std(stacked_logits, dim=0)
        weights = 1 / (1 + stddev_logits)
        weighted_mean_logits = torch.sum(weights * stacked_logits, dim=0) / torch.sum(weights, dim=0)
        alpha = self.alpha
        lm_logits = alpha * mean_logits + (1 - alpha) * weighted_mean_logits
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
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