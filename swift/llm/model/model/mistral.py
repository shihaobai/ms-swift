# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from transformers import AutoProcessor, AutoTokenizer

from swift.llm import TemplateType
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal,
                        get_model_tokenizer_with_flash_attn, register_model)
from ..utils import ModelInfo, safe_snapshot_download
import torch
from typing import Optional, Union
from transformers import (
    MistralConfig,
    MistralForCausalLM,
    GradientCheckpointingLayer,
)
from transformers.cache_utils import Cache
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.models.mistral.modeling_mistral import MistralMLP, MistralRMSNorm
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
import torch
import torch.nn as nn

register_model(
    ModelMeta(
        LLMModelType.mistral,
        [
            ModelGroup([
                Model('AI-ModelScope/Mistral-7B-Instruct-v0.1', 'mistralai/Mistral-7B-Instruct-v0.1'),
                Model('AI-ModelScope/Mistral-7B-Instruct-v0.2', 'mistralai/Mistral-7B-Instruct-v0.2'),
                Model('LLM-Research/Mistral-7B-Instruct-v0.3', 'mistralai/Mistral-7B-Instruct-v0.3'),
                Model('AI-ModelScope/Mistral-7B-v0.1', 'mistralai/Mistral-7B-v0.1'),
                Model('AI-ModelScope/Mistral-7B-v0.2-hf', 'alpindale/Mistral-7B-v0.2-hf'),
            ]),
            ModelGroup([
                Model('swift/Codestral-22B-v0.1', 'mistralai/Codestral-22B-v0.1'),
            ]),
        ],
        TemplateType.llama,
        get_model_tokenizer_with_flash_attn,
        architectures=['MistralForCausalLM'],
        model_arch=ModelArch.llama,
        requires=['transformers>=4.34'],
    ))

register_model(
    ModelMeta(
        LLMModelType.mixtral, [
            ModelGroup([
                Model('AI-ModelScope/Mixtral-8x7B-Instruct-v0.1', 'mistralai/Mixtral-8x7B-Instruct-v0.1'),
                Model('AI-ModelScope/Mixtral-8x7B-v0.1', 'mistralai/Mixtral-8x7B-v0.1'),
                Model('AI-ModelScope/Mixtral-8x22B-v0.1', 'mistral-community/Mixtral-8x22B-v0.1'),
            ],
                       requires=['transformers>=4.36']),
            ModelGroup([
                Model('AI-ModelScope/Mixtral-8x7b-AQLM-2Bit-1x16-hf', 'ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf'),
            ],
                       requires=['transformers>=4.38', 'aqlm', 'torch>=2.2.0']),
        ],
        TemplateType.llama,
        get_model_tokenizer_with_flash_attn,
        architectures=['MixtralForCausalLM'],
        model_arch=ModelArch.llama))

register_model(
    ModelMeta(
        LLMModelType.mistral_nemo, [
            ModelGroup([
                Model('AI-ModelScope/Mistral-Small-Instruct-2409', 'mistralai/Mistral-Small-Instruct-2409'),
                Model('LLM-Research/Mistral-Large-Instruct-2407', 'mistralai/Mistral-Large-Instruct-2407'),
                Model('AI-ModelScope/Mistral-Nemo-Base-2407', 'mistralai/Mistral-Nemo-Base-2407'),
                Model('AI-ModelScope/Mistral-Nemo-Instruct-2407', 'mistralai/Mistral-Nemo-Instruct-2407'),
            ],
                       requires=['transformers>=4.43']),
            ModelGroup([
                Model('AI-ModelScope/Ministral-8B-Instruct-2410', 'mistralai/Ministral-8B-Instruct-2410'),
            ],
                       requires=['transformers>=4.46']),
        ],
        TemplateType.mistral_nemo,
        get_model_tokenizer_with_flash_attn,
        architectures=['MistralForCausalLM'],
        model_arch=ModelArch.llama))

register_model(
    ModelMeta(
        LLMModelType.mistral_2501, [
            ModelGroup([
                Model('mistralai/Mistral-Small-24B-Base-2501', 'mistralai/Mistral-Small-24B-Base-2501'),
                Model('mistralai/Mistral-Small-24B-Instruct-2501', 'mistralai/Mistral-Small-24B-Instruct-2501'),
            ]),
        ],
        TemplateType.mistral_2501,
        get_model_tokenizer_with_flash_attn,
        architectures=['MistralForCausalLM'],
        model_arch=ModelArch.llama))

register_model(
    ModelMeta(
        LLMModelType.zephyr,
        [
            ModelGroup([
                Model('modelscope/zephyr-7b-beta', 'HuggingFaceH4/zephyr-7b-beta'),
            ]),
        ],
        TemplateType.zephyr,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['MistralForCausalLM'],
        requires=['transformers>=4.34'],
    ))

register_model(
    ModelMeta(
        LLMModelType.wizardlm2_moe,
        [ModelGroup([
            Model('AI-ModelScope/WizardLM-2-8x22B', 'alpindale/WizardLM-2-8x22B'),
        ])],
        TemplateType.wizardlm2_moe,
        get_model_tokenizer_with_flash_attn,
        architectures=['MixtralForCausalLM'],
        requires=['transformers>=4.36'],
    ))

register_model(
    ModelMeta(
        LLMModelType.wizardlm2,
        [ModelGroup([
            Model('AI-ModelScope/WizardLM-2-7B-AWQ', 'MaziyarPanahi/WizardLM-2-7B-AWQ'),
        ])],
        TemplateType.wizardlm2,
        get_model_tokenizer_with_flash_attn,
        architectures=['MistralForCausalLM'],
        requires=['transformers>=4.34'],
    ))


def get_model_tokenizer_mistral_2503(model_dir: str,
                                     model_info: ModelInfo,
                                     model_kwargs: Dict[str, Any],
                                     load_model: bool = True,
                                     **kwargs):
    from transformers import Mistral3ForConditionalGeneration
    kwargs['automodel_class'] = kwargs['automodel_class'] or Mistral3ForConditionalGeneration
    model, processor = get_model_tokenizer_multimodal(model_dir, model_info, model_kwargs, load_model, **kwargs)

    return model, processor


def get_model_tokenizer_devstral_2505(model_dir: str,
                                      model_info: ModelInfo,
                                      model_kwargs: Dict[str, Any],
                                      load_model: bool = True,
                                      **kwargs):
    # src: sglang did the same (https://github.com/sgl-project/sglang/pull/6547)
    tokenizer_dir = safe_snapshot_download('mistralai/Mistral-Small-3.1-24B-Instruct-2503', download_model=False)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    kwargs['tokenizer'] = tokenizer
    model, processor = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    return model, processor


register_model(
    ModelMeta(
        model_type=LLMModelType.devstral,
        model_groups=[
            ModelGroup([
                Model('mistralai/Devstral-Small-2505', 'mistralai/Devstral-Small-2505'),
            ],
                       requires=['transformers>=4.43', 'mistral-common>=1.5.5'])
        ],
        template=TemplateType.devstral,
        get_function=get_model_tokenizer_devstral_2505,
        architectures=['MistralForCausalLM'],
        model_arch=ModelArch.llama))

register_model(
    ModelMeta(
        MLLMModelType.mistral_2503,
        [
            ModelGroup([
                Model('mistralai/Mistral-Small-3.1-24B-Base-2503', 'mistralai/Mistral-Small-3.1-24B-Base-2503'),
                Model('mistralai/Mistral-Small-3.1-24B-Instruct-2503', 'mistralai/Mistral-Small-3.1-24B-Instruct-2503'),
            ]),
        ],
        TemplateType.mistral_2503,
        get_model_tokenizer_mistral_2503,
        architectures=['Mistral3ForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.49'],
    ))


def get_model_tokenizer_mistral_2506(model_dir: str,
                                     model_info: ModelInfo,
                                     model_kwargs: Dict[str, Any],
                                     load_model: bool = True,
                                     **kwargs):
    from transformers import Mistral3ForConditionalGeneration
    tokenizer_dir = safe_snapshot_download('mistralai/Mistral-Small-3.1-24B-Instruct-2503', download_model=False)
    processor = AutoProcessor.from_pretrained(tokenizer_dir)
    kwargs['automodel_class'] = kwargs['automodel_class'] or Mistral3ForConditionalGeneration
    kwargs['tokenizer'] = processor.tokenizer
    model, _ = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.mistral_2506,
        [
            ModelGroup([
                Model('mistralai/Mistral-Small-3.2-24B-Instruct-2506', 'mistralai/Mistral-Small-3.2-24B-Instruct-2506'),
            ]),
        ],
        TemplateType.mistral_2506,
        get_model_tokenizer_mistral_2506,
        architectures=['Mistral3ForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.49'],
    ))


def get_model_tokenizer_mistral_2512(model_dir: str,
                                     model_info: ModelInfo,
                                     model_kwargs: Dict[str, Any],
                                     load_model: bool = True,
                                     **kwargs):
    from transformers import Mistral3ForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_dir)
    kwargs['automodel_class'] = kwargs['automodel_class'] or Mistral3ForConditionalGeneration
    kwargs['tokenizer'] = processor.tokenizer
    model, _ = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.mistral_2512,
        [
            ModelGroup([
                Model('mistralai/Ministral-3-3B-Base-2512', 'mistralai/Ministral-3-3B-Base-2512'),
                Model('mistralai/Ministral-3-3B-Instruct-2512', 'mistralai/Ministral-3-3B-Instruct-2512'),
                Model('mistralai/Ministral-3-3B-Instruct-2512-BF16', 'mistralai/Ministral-3-3B-Instruct-2512-BF16'),
                Model('mistralai/Ministral-3-8B-Base-2512', 'mistralai/Ministral-3-8B-Base-2512'),
                Model('mistralai/Ministral-3-8B-Instruct-2512', 'mistralai/Ministral-3-8B-Instruct-2512'),
                Model('mistralai/Ministral-3-8B-Instruct-2512-BF16', 'mistralai/Ministral-3-8B-Instruct-2512-BF16'),
                Model('mistralai/Ministral-3-14B-Base-2512', 'mistralai/Ministral-3-14B-Base-2512'),
                Model('mistralai/Ministral-3-14B-Instruct-2512', 'mistralai/Ministral-3-14B-Instruct-2512'),
                Model('mistralai/Ministral-3-14B-Instruct-2512-BF16', 'mistralai/Ministral-3-14B-Instruct-2512-BF16'),
            ]),
        ],
        TemplateType.mistral_2512,
        get_model_tokenizer_mistral_2512,
        architectures=['Mistral3ForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=5.0.0.dev0', 'mistral-common>=1.8.6'],
        tags=['vision'],
        ignore_patterns=[],
    ))

register_model(
    ModelMeta(
        MLLMModelType.mistral_2512_thinking,
        [
            ModelGroup([
                Model('mistralai/Ministral-3-3B-Reasoning-2512', 'mistralai/Ministral-3-3B-Reasoning-2512'),
                Model('mistralai/Ministral-3-8B-Reasoning-2512', 'mistralai/Ministral-3-8B-Reasoning-2512'),
                Model('mistralai/Ministral-3-14B-Reasoning-2512', 'mistralai/Ministral-3-14B-Reasoning-2512'),
            ]),
        ],
        TemplateType.mistral_2512_thinking,
        get_model_tokenizer_mistral_2512,
        architectures=['Mistral3ForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=5.0.0.dev0', 'mistral-common>=1.8.6'],
        tags=['vision'],
        ignore_patterns=[],
    ))

class MistralMTPBlock(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MistralMLP(config)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class MistralMTP(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        mtp_num_layers = config.mtp_num_layers if hasattr(config, 'mtp_num_layers') else 1
        self.enorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.layers = nn.ModuleList([MistralMTPBlock(config) for _ in range(mtp_num_layers)])
    
    def forward(
        self,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )

        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

class MistralForCausalLMWithMTP(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.mtp = MistralMTP(config)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, MistralForCausalLM

        >>> model = MistralForCausalLM.from_pretrained("meta-mistral/Mistral-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-mistral/Mistral-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        with torch.no_grad():
            outputs: BaseModelOutputWithPast = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        # logits = self.lm_head(hidden_states[:, slice_indices, :])
        # import ipdb; ipdb.set_trace()
        # input_ids: [batch_size, seq_len]
        # logits: [batch_size, seq_len, vocab_size]
        # labels: [batch_size, seq_len]
        # labels的 -100 是mask的含义
        with torch.no_grad():
            loss = None
            input_ids = input_ids[:, 1:].contiguous()
            hidden_states = hidden_states[:, :-1]
            labels = labels[:, 1:].contiguous()
            inputs_embeds = self.model.embed_tokens(input_ids)
        hidden_states = hidden_states.requires_grad_(True)
        inputs_embeds = inputs_embeds.requires_grad_(True)
        mtp_hidden_states = self.mtp(previous_hidden_states=hidden_states, inputs_embeds=inputs_embeds)
        mtp_hidden_states = self.model.norm(mtp_hidden_states)
        mtp_logits = self.lm_head(mtp_hidden_states)
        # print(mtp_logits.shape, labels.shape, labels.max(), labels.min())
        if labels is not None:
            loss = self.loss_function(logits=mtp_logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=mtp_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=mtp_hidden_states,
            attentions=outputs.attentions,
        )

def get_model_tokenizer_mistral_with_mtp(model_dir: str,
                                  model_info: ModelInfo,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  **kwargs):
    kwargs['automodel_class'] = MistralForCausalLMWithMTP
    return get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)

register_model(
    ModelMeta(
        model_type=LLMModelType.mistral_with_mtp,            # 在 constant.py 中先加枚举
        model_groups=[
            ModelGroup([
                Model('AI-ModelScope/Mistral-7B-Instruct-v0.1', 'mistralai/Mistral-7B-Instruct-v0.1'),
                Model('AI-ModelScope/Mistral-7B-Instruct-v0.2', 'mistralai/Mistral-7B-Instruct-v0.2'),
                Model('LLM-Research/Mistral-7B-Instruct-v0.3', 'mistralai/Mistral-7B-Instruct-v0.3'),
                Model('AI-ModelScope/Mistral-7B-v0.1', 'mistralai/Mistral-7B-v0.1'),
                Model('AI-ModelScope/Mistral-7B-v0.2-hf', 'alpindale/Mistral-7B-v0.2-hf'),
            ]),
        ],
        template=TemplateType.mistral_niren,           
        get_function=get_model_tokenizer_mistral_with_mtp,  
        architectures=['MistralForCausalLMWithMTP'],       
        model_arch=ModelArch.llama,                   
        requires=['transformers>=4.49']               
    ))