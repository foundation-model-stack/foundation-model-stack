import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fms import models
from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.models.conformer import ConformerConfig, ConformerEncoder
from fms.models.granite import GraniteConfig, GraniteHeadless
from fms.modules.projector import SpeechProjector, SpeechProjectorConfig
from fms.utils.config import ModelConfig

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

try:
    from peft import PeftModel  # noqa: F401
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def is_peft_available() -> bool:
    return PEFT_AVAILABLE


logger = logging.getLogger(__name__)


_default_encoder_config = ConformerConfig(
    num_features=160,
    hidden_dim=1024,
    num_layers=16,
    num_heads=8,
    dim_head=128,
    conv_kernel_size=15,
    conv_expansion_factor=2,
    feedforward_mult=4,
    dropout=0.1,
    max_pos_emb=512,
    context_size=200,
    output_dim=256,
)

_default_projector_config = SpeechProjectorConfig(
    encoder_dim=1024,
    decoder_dim=4096,
    num_hidden_layers=2,
    num_attention_heads=16,
    intermediate_size=4096,
    hidden_dropout_prob=0.1,
    attention_dropout_prob=0.1,
    hidden_act="gelu",
    layer_norm_eps=1e-12,
    initializer_range=0.02,
)

_default_decoder_config = GraniteConfig(
    src_vocab_size=49160,
    emb_dim=4096,
    norm_eps=1e-5,
    nheads=32,
    head_dim=128,
    kvheads=8,
    nlayers=40,
    hidden_grow_factor=12800 / 4096,
    max_expected_seq_len=131072,
    rope_theta=10000000.0,
    pad_id=0,
    p_dropout=0.0,
    tie_heads=True,
    fused_weights=True,
    residual_multiplier=0.22,
    embedding_multiplier=12.0,
    attention_multiplier=0.0078125,
    logits_scaling=16.0,
)


@dataclass
class GraniteSpeechConfig(ModelConfig):
    """Configuration for Granite Speech multimodal model (Conformer encoder + Granite decoder)."""
    encoder_config: ConformerConfig = field(default_factory=lambda: _default_encoder_config)
    projector_config: SpeechProjectorConfig = field(default_factory=lambda: _default_projector_config)
    decoder_config: GraniteConfig = field(default_factory=lambda: _default_decoder_config)
    audio_token_index: int = 49159  # Token ID used as placeholder for audio embeddings
    has_lora_adapter: bool = True
    downsample_rate: int = 5  # Temporal downsampling factor in projector
    window_size: int = 15  # Context window size for Q-Former projector
    initializer_range: float = 0.02
    freeze_encoder: bool = False
    freeze_decoder: bool = False


class GraniteSpeech(nn.Module):
    """Multimodal speech-to-text model combining Conformer encoder with Granite decoder."""

    def __init__(
        self,
        config: Optional[GraniteSpeechConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(GraniteSpeech, self).__init__()

        if config is not None:
            self.config = config
        else:
            self.config = GraniteSpeechConfig()

        self.config = self.config.updated(**kwargs) if kwargs else self.config
        self.distributed_strategy = distributed_strategy
        self._peft_adapter_loaded = False

        self.audio_token_index = self.config.audio_token_index
        self.window_size = self.config.window_size
        self.downsample_rate = self.config.downsample_rate

        # Encoder-projector-decoder pipeline for speech-to-text
        self.encoder = ConformerEncoder(self.config.encoder_config)
        self.projector = SpeechProjector(
            self.config.projector_config,
            window_size=self.config.window_size,
            downsample_rate=self.config.downsample_rate,
        )
        self.decoder = GraniteHeadless(self.config.decoder_config)
        self.lm_head = nn.Linear(
            self.config.decoder_config.emb_dim,
            self.config.decoder_config.src_vocab_size,
            bias=False,
        )

        # Freeze encoder/decoder if specified (for finetuning)
        if self.config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.config.freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.lm_head.parameters():
                param.requires_grad = False

        # NOTE: LoRA adapters only applied to decoder attention layers
        if self.config.has_lora_adapter and not is_peft_available():
            logger.warning(
                "Config indicates that a lora adapter should be present, but "
                "peft is not installed; this will cause the model to perform "
                "incorrectly when audio inputs are provided. Please install "
                "peft and reload the model!"
            )

    @classmethod
    def from_config(cls, config: GraniteSpeechConfig) -> "GraniteSpeech":
        return cls(config)

    def load_adapter(self, adapter_path: str):
        if not is_peft_available():
            raise ImportError("peft is required to load LoRA adapters. Please install peft.")

        from peft import PeftModel

        self.decoder = PeftModel.from_pretrained(self.decoder, adapter_path)
        self._peft_adapter_loaded = True

    def get_config(self) -> GraniteSpeechConfig:
        return self.config

    def _maybe_toggle_adapters(self, input_features: Optional[torch.Tensor]):
        if not (is_peft_available() and self._peft_adapter_loaded):
            return

        if input_features is not None:
            self.decoder.enable_adapters()
        else:
            self.decoder.disable_adapters()

    @staticmethod
    def _fix_state_dict_key_on_save(key: str) -> Tuple[str, bool]:
        return key.replace(".base_layer", ""), False

    def _fix_state_dict_keys_on_save(self, state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        if is_peft_available() and self._peft_adapter_loaded:
            return state_dict

        fixed = {}
        for key, value in state_dict.items():
            if ".lora_" in key:
                continue
            new_key, _ = self._fix_state_dict_key_on_save(key)
            fixed[new_key] = value
        return fixed

    def _get_adapter_name(self) -> str:
        if not hasattr(self.decoder, "peft_config"):
            raise ValueError("Decoder does not have PEFT adapters loaded.")
        return list(self.decoder.peft_config.keys())[0]

    def reset_parameters(self):
        nn.init.normal_(self.lm_head.weight, std=self.config.initializer_range)
        self.decoder.reset_parameters()

    def post_init(self):
        self.encoder._recompute_buffers()

        if self.config.decoder_config.tie_heads:
            # Always tie lm_head to embedding, not the other way around.
            # When loading from HF checkpoint with tie_word_embeddings=True,
            # lm_head.weight is not in the checkpoint, so embedding has the
            # correct weights and lm_head should point to it.
            self.lm_head.weight = self.decoder.embedding.weight

    def get_input_embeddings(self):
        return self.decoder.embedding

    def get_output_embeddings(self):
        return self.lm_head

    def get_audio_features(self, input_features: torch.Tensor) -> torch.Tensor:
        input_features = input_features.to(dtype=self.encoder.input_proj.weight.dtype)
        encoder_embeds = self.encoder(input_features)
        projected_embeds = self.projector(encoder_embeds)
        return projected_embeds

    def get_merged_audio_embeddings(
        self,
        input_ids: torch.Tensor,
        audio_features: torch.Tensor,
        input_features_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Replace audio token placeholders with actual audio embeddings
        audio_pos = (input_ids == self.audio_token_index)
        safe_ids = torch.where(audio_pos, input_ids.new_zeros(()), input_ids)
        token_embeds = self.get_input_embeddings()(safe_ids)

        audio_features = audio_features.to(token_embeds.device, token_embeds.dtype)
        audio_mask = input_features_mask
        if audio_mask.shape != audio_features.shape[:2]:
            audio_mask = torch.ones(
                audio_features.shape[:2],
                device=audio_features.device,
                dtype=torch.bool,
            )
        audio_flat = audio_features[audio_mask]

        # Verify audio token count matches provided audio embeddings
        expected = audio_pos.sum().item()
        if expected * token_embeds.size(-1) != audio_flat.numel():
            raise ValueError(
                f"Mismatch: {expected} audio positions but "
                f"{audio_flat.shape[0]} audio vectors provided."
            )

        # Scatter audio embeddings into token positions
        mask = audio_pos.unsqueeze(-1)
        merged = token_embeds.masked_scatter(mask, audio_flat)

        return merged

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        input_features_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[Tuple] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        # Handle past_key_value_states alias
        if past_key_values is None and "past_key_value_states" in kwargs:
            past_key_values = kwargs.pop("past_key_value_states")

        # Handle inputs_embeds from generation hook
        if inputs_embeds is None and "inputs_embeds" in kwargs:
            inputs_embeds = kwargs.pop("inputs_embeds")
            if input_ids is not None:
                input_ids = None

        if input_ids is None and inputs_embeds is None:
            raise ValueError("Specify input_ids or inputs_embeds.")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("input_ids and inputs_embeds are mutually exclusive.")
        if input_features is not None and inputs_embeds is not None:
            raise ValueError("input_features and inputs_embeds cannot be used together.")

        if inputs_embeds is None:
            has_audio_token = input_ids is not None and (input_ids == self.audio_token_index).any()

            # Process audio+text multimodal input
            if input_features is not None and has_audio_token:
                if input_ids is None:
                    raise ValueError("input_ids are required when using input_features.")
                audio_embeds = self.get_audio_features(input_features)

                if input_features_mask is None:
                    input_features_mask = audio_embeds.new_ones(
                        audio_embeds.shape[:2], dtype=torch.bool
                    )
                input_features_mask = input_features_mask.to(
                    device=audio_embeds.device, dtype=torch.bool
                )

                inputs_embeds = self.get_merged_audio_embeddings(
                    input_ids=input_ids,
                    audio_features=audio_embeds,
                    input_features_mask=input_features_mask,
                )
            else:
                # Text-only input
                inputs_embeds = self.get_input_embeddings()(input_ids)

        dec_out, cache = self.decoder(
            x_in=inputs_embeds,
            position_ids=position_ids,
            past_key_value_states=past_key_values,
            attention_mask=attention_mask,
            use_cache=bool(use_cache),
        )

        logits = self.lm_head(dec_out)
        logits = logits / self.config.decoder_config.logits_scaling

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            shift_logits = torch.nan_to_num(shift_logits)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if use_cache:
            return logits, cache

        return logits, loss

    def prepare_inputs_for_generation(
        self,
        iteration: int,
        input_ids: torch.Tensor,
        kwargs: dict,
    ) -> Tuple[torch.Tensor, dict]:
        # After first iteration, use cached key-values and skip embedding computation
        if kwargs.get("use_cache", False) and iteration > 0:
            kwargs.pop("inputs_embeds", None)
            return input_ids, kwargs

        input_features = kwargs.pop("input_features", None)
        input_features_mask = kwargs.pop("input_features_mask", None)

        # Enable LoRA adapters only when processing audio inputs
        if iteration == 0:
            self._maybe_toggle_adapters(input_features)

        if input_features is None:
            return input_ids, kwargs

        audio_embeds = self.get_audio_features(input_features)

        if input_features_mask is None:
            input_features_mask = audio_embeds.new_ones(
                audio_embeds.shape[:2], dtype=torch.bool
            )
        input_features_mask = input_features_mask.to(
            device=audio_embeds.device, dtype=torch.bool
        )

        inputs_embeds = self.get_merged_audio_embeddings(
            input_ids=input_ids,
            audio_features=audio_embeds,
            input_features_mask=input_features_mask,
        )

        kwargs["inputs_embeds"] = inputs_embeds
        return None, kwargs


def _granite_speech_factory_factory(config: GraniteSpeechConfig):
    def factory(**kwargs):
        return GraniteSpeech(config, **kwargs)
    return factory


_architecture_name = "granite_speech"
_granite_speech_default = GraniteSpeechConfig()

_granite_speech_2b = GraniteSpeechConfig(
    encoder_config=_default_encoder_config.updated(output_dim=256),
    projector_config=_default_projector_config.updated(decoder_dim=2048),
    decoder_config=GraniteConfig(
        src_vocab_size=49160,
        emb_dim=2048,
        norm_eps=1e-5,
        nheads=32,
        head_dim=64,
        kvheads=8,
        nlayers=40,
        hidden_grow_factor=8192 / 2048,
        max_expected_seq_len=8192,
        rope_theta=10000000.0,
        pad_id=0,
        p_dropout=0.0,
        tie_heads=True,
        fused_weights=True,
        residual_multiplier=0.22,
        embedding_multiplier=12.0,
        attention_multiplier=0.015625,
        logits_scaling=8.0,
    ),
)

models.register_model(
    _architecture_name, "3.3-8b", _granite_speech_factory_factory(_granite_speech_default)
)
models.register_model(
    _architecture_name, "3.2-8b", _granite_speech_factory_factory(_granite_speech_default)
)
models.register_model(
    _architecture_name, "3.3-2b", _granite_speech_factory_factory(_granite_speech_2b)
)


def _merge_lora_weights(
    input_sd: Mapping[str, Any], **kwargs
) -> Mapping[str, Any]:
    """Merges LoRA adapter weights into base model weights for deployment."""
    new_sd = {}
    lora_keys_processed = set()
    lora_scaling = 0.5

    lora_a_pattern = re.compile(
        r"^(decoder\.layers\.\d+\.attn\.in_proj\.(query|value))\.lora_A(?:\.default)?\.weight$"
    )

    for name, param in input_sd.items():
        match = lora_a_pattern.match(name)
        if match:
            base_key = match.group(1)
            proj_type = match.group(2)

            lora_a_key = name
            lora_b_key = name.replace("lora_A", "lora_B")
            base_weight_key = f"{base_key}.weight"

            if lora_b_key in input_sd and base_weight_key in input_sd:
                lora_a = input_sd[lora_a_key]
                lora_b = input_sd[lora_b_key]
                base_weight = input_sd[base_weight_key]

                lora_delta = torch.matmul(lora_b, lora_a) * lora_scaling
                merged_weight = base_weight + lora_delta

                new_sd[base_weight_key] = merged_weight
                lora_keys_processed.add(lora_a_key)
                lora_keys_processed.add(lora_b_key)
                lora_keys_processed.add(base_weight_key)

                logger.debug(
                    f"Merged LoRA weights for {base_key}: "
                    f"base={base_weight.shape}, lora_A={lora_a.shape}, lora_B={lora_b.shape}"
                )

    for name, param in input_sd.items():
        if name not in lora_keys_processed:
            if ".lora_A." in name or ".lora_B." in name:
                logger.warning(f"Skipping orphaned LoRA key: {name}")
                continue
            new_sd[name] = param

    lora_merged_count = len(lora_keys_processed) // 3
    if lora_merged_count > 0:
        logger.info(f"Merged {lora_merged_count} LoRA adapter pairs into base weights")

    return new_sd


def _hf_to_fms_names(
    input_sd: Mapping[str, Any], **kwargs
) -> Mapping[str, Any]:
    """Converts HuggingFace checkpoint keys to FMS naming convention."""
    encoder_replacements = [
        (r"^encoder\.input_linear", "encoder.input_proj"),
        (r"^encoder\.layers\.(\d+)", r"encoder.blocks.\1"),
        (r"\.ff1\.pre_norm\.", ".ff1.norm."),
        (r"\.ff1\.up_proj\.", ".ff1.fc1."),
        (r"\.ff1\.down_proj\.", ".ff1.fc2."),
        (r"\.attn\.pre_norm\.", ".attn.norm."),
        (r"\.attn\.to_kv\.", ".attn.to_kv."),
        (r"\.attn\.rel_pos_emb\.", ".attn.pos_emb."),
        (r"\.conv\.up_conv\.", ".conv.pointwise_conv1."),
        (r"\.conv\.depth_conv\.conv\.", ".conv.depthwise_conv."),
        (r"\.conv\.down_conv\.", ".conv.pointwise_conv2."),
        (r"\.ff2\.pre_norm\.", ".ff2.norm."),
        (r"\.ff2\.up_proj\.", ".ff2.fc1."),
        (r"\.ff2\.down_proj\.", ".ff2.fc2."),
    ]

    projector_replacements = [
        (r"^projector\.query$", "projector.query_embeds"),
        (r"^projector\.qformer\.layernorm\.", "projector.input_layernorm."),
        (r"^projector\.qformer\.encoder\.layer\.(\d+)", r"projector.layers.\1"),
        (r"\.attention\.attention\.query\.", ".self_attention.query."),
        (r"\.attention\.attention\.key\.", ".self_attention.key."),
        (r"\.attention\.attention\.value\.", ".self_attention.value."),
        (r"\.attention\.output\.dense\.", ".self_attention_output.dense."),
        (r"\.attention\.output\.LayerNorm\.", ".self_attention_output.LayerNorm."),
        (r"\.crossattention\.attention\.query\.", ".cross_attention.query."),
        (r"\.crossattention\.attention\.key\.", ".cross_attention.key."),
        (r"\.crossattention\.attention\.value\.", ".cross_attention.value."),
        (r"\.crossattention\.output\.dense\.", ".cross_attention_output.dense."),
        (r"\.crossattention\.output\.LayerNorm\.", ".cross_attention_output.LayerNorm."),
        (r"\.intermediate_query\.dense\.", ".intermediate_query.dense."),
        (r"\.output_query\.dense\.", ".output_query.dense."),
        (r"\.output_query\.LayerNorm\.", ".output_query.LayerNorm."),
        (r"^projector\.linear\.", "projector.output_proj."),
    ]

    decoder_replacements = [
        (r"^language_model\.lm_head\.weight", "lm_head.weight"),
        (r"^language_model\.model\.embed_tokens\.weight", "decoder.embedding.weight"),
        (r"^language_model\.model\.norm", "decoder.dec_norm"),
        (r"^language_model\.model\.layers", "decoder.layers"),
        (r"self_attn\.k_proj", "attn.in_proj.key"),
        (r"self_attn\.v_proj", "attn.in_proj.value"),
        (r"self_attn\.q_proj", "attn.in_proj.query"),
        (r"self_attn\.o_proj", "attn.dense"),
        (r"mlp\.gate_proj", "ff_sub_layer.wg"),
        (r"mlp\.up_proj", "ff_sub_layer.w1"),
        (r"mlp\.down_proj", "ff_sub_layer.w2"),
        (r"(decoder\.layers\.\d+\.)input_layernorm", r"\1ln"),
        (r"(decoder\.layers\.\d+\.)post_attention_layernorm", r"\1ff_ln"),
    ]

    all_replacements = encoder_replacements + projector_replacements + decoder_replacements

    new_sd = {}
    for name, param in input_sd.items():
        new_name = name
        for pattern, repl in all_replacements:
            new_name = re.sub(pattern, repl, new_name)
        if ".base_layer." in new_name and ".lora_" not in new_name:
            new_name = new_name.replace(".base_layer", "")
        new_sd[new_name] = param

    return new_sd


def _is_peft_key(key: str) -> bool:
    return ".lora_" in key


def _granite_speech_weight_fusion(
    input_sd: Mapping[str, Any],
    model_config: Optional[GraniteSpeechConfig] = None,
    **kwargs
) -> Mapping[str, Any]:
    """Fuses attention and MLP weights for improved inference efficiency."""
    from fms.utils import serialization

    decoder_sd = {k: v for k, v in input_sd.items()
                  if k.startswith("decoder.") and not _is_peft_key(k)}
    peft_sd = {k: v for k, v in input_sd.items() if _is_peft_key(k)}
    other_sd = {k: v for k, v in input_sd.items()
                if not k.startswith("decoder.") and not _is_peft_key(k)}

    has_fused_weights = True
    if model_config and model_config.decoder_config:
        if not model_config.decoder_config.fused_weights:
            has_fused_weights = False

    if has_fused_weights and decoder_sd:
        decoder_sd = serialization._mlp_glu_unfused_to_fused_adapter_step(
            serialization._attn_unfused_to_fused_step(decoder_sd)
        )

    new_sd = {**other_sd, **decoder_sd, **peft_sd}
    return new_sd


def _hf_to_fms_rope(
    input_sd: Mapping[str, Any], model_config: Optional[GraniteSpeechConfig] = None, **kwargs
) -> Mapping[str, Any]:
    """Applies RoPE (Rotary Position Embedding) weight transformations for query/key projections."""
    new_sd = {}

    if model_config:
        head_size = model_config.decoder_config.head_dim
    else:
        logger.warning("Missing model_config, assuming default head_size=64")
        head_size = 64

    rope_pattern = re.compile(
        r"^decoder\.layers\.\d+\.attn\.in_proj\.(query|key)\.weight$"
    )

    for name, param in input_sd.items():
        if rope_pattern.match(name) and param.numel() > 1:
            # Interleave RoPE dimensions: (h, d/2, 2) -> (h, 2, d/2)
            temp = param
            num_heads = temp.size(0) // head_size

            if temp.dim() == 2:
                temp_view = temp.view(num_heads, 2, -1, temp.size(1))
            else:
                temp_view = temp.view(num_heads, 2, -1)
            temp = temp_view.transpose(1, 2).reshape(*param.size())

            new_sd[name] = temp
            logger.debug(f"Applied RoPE transformation to {name}")
        else:
            new_sd[name] = param

    return new_sd


try:
    from fms.utils import serialization

    serialization.register_adapter_step(
        _architecture_name, "hf_to_fms_names", _hf_to_fms_names
    )

    serialization.register_adapter_step(
        _architecture_name, "merge_lora_weights", _merge_lora_weights
    )

    serialization.register_adapter_step(
        _architecture_name, "hf_to_fms_rope", _hf_to_fms_rope
    )

    serialization.register_adapter_step(
        _architecture_name, "weight_fusion", _granite_speech_weight_fusion
    )

    serialization.register_adapter(
        _architecture_name,
        "hf",
        [
            "hf_to_fms_names",
            "merge_lora_weights",
            "hf_to_fms_rope",
            "weight_fusion",
        ],
    )
except ImportError:
    logger.warning("Could not register serialization adapters - fms.serialization not available")


class GraniteSpeechFeatureExtractor:
    """Extracts mel-spectrogram features from raw audio for speech processing."""

    def __init__(
        self,
        sampling_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,  # Number of mel filterbanks
        projector_window_size: int = 15,
        projector_downsample_rate: int = 5,
        **kwargs,
    ):
        if not TORCHAUDIO_AVAILABLE:
            raise ImportError(
                "torchaudio is required for GraniteSpeechFeatureExtractor. "
                "Please install it with: pip install torchaudio"
            )

        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.projector_window_size = projector_window_size
        self.projector_downsample_rate = projector_downsample_rate

        self.melspec_kwargs = {
            "sample_rate": sampling_rate,
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
            "n_mels": n_mels,
        }
        self.mel_filters = torchaudio.transforms.MelSpectrogram(**self.melspec_kwargs)

    def __call__(
        self,
        audios: Union[torch.Tensor, Sequence[torch.Tensor], np.ndarray, Sequence[np.ndarray]],
        device: Optional[str] = "cpu",
        **kwargs,
    ) -> dict:
        speech_inputs = {}

        batched_audio, audio_lengths = self._get_audios_and_audio_lengths(audios)
        speech_inputs["input_features"] = self._extract_mel_spectrograms(
            batched_audio,
            device=device,
        )

        audio_embed_sizes = self._get_num_audio_features(audio_lengths)
        speech_inputs["audio_embed_sizes"] = audio_embed_sizes

        speech_inputs["input_features_mask"] = torch.arange(max(audio_embed_sizes)).view(1, -1) < torch.tensor(
            audio_embed_sizes
        ).view(-1, 1)

        return speech_inputs

    def _extract_mel_spectrograms(
        self,
        audio: torch.Tensor,
        device: str = "cpu"
    ) -> torch.Tensor:
        if device is not None:
            melspec = self.mel_filters.to(device)
            audio = audio.to(device)
        else:
            melspec = self.mel_filters

        bsz = audio.shape[0]

        with torch.no_grad():
            # Compute mel-spectrogram and apply log normalization
            mel = melspec(audio.float())
            logmel = mel.transpose(-1, -2).clip_(min=1e-10).log10_()
            mx = logmel.amax(dim=(-2, -1), keepdim=True)
            logmel = torch.maximum(logmel, mx - 8.0).div_(4).add_(1)

            # Ensure even length for pairwise stacking
            if logmel.shape[1] % 2 == 1:
                logmel = logmel[:, :-1]

            # Stack adjacent frames to create 2-channel features
            audio_features = logmel.reshape(bsz, -1, 2 * logmel.shape[-1])

        return audio_features

    def _get_num_audio_features(
        self,
        audio_lengths: Sequence[int]
    ) -> list[int]:
        effective_window_size = self.projector_window_size // self.projector_downsample_rate

        projector_lengths = []
        for raw_length in audio_lengths:
            mel_length = raw_length // self.hop_length + 1
            encoder_length = mel_length // 2
            nblocks = math.ceil(encoder_length / self.projector_window_size)
            projector_length = nblocks * effective_window_size
            projector_lengths.append(projector_length)

        return projector_lengths

    def _get_audios_and_audio_lengths(
        self,
        audios: Union[torch.Tensor, Sequence[torch.Tensor], np.ndarray, Sequence[np.ndarray]]
    ) -> Tuple[torch.Tensor, list[int]]:
        if isinstance(audios, np.ndarray):
            audios = torch.from_numpy(audios)
        elif isinstance(audios, Sequence) and len(audios) > 0 and isinstance(audios[0], np.ndarray):
            audios = [torch.from_numpy(arr) for arr in audios]

        if isinstance(audios, torch.Tensor):
            if audios.ndim == 1:
                audios = audios.unsqueeze(0)
            if not torch.is_floating_point(audios):
                raise ValueError("Invalid audio provided. Audio should be a floating point tensor")

            if audios.shape[0] > 1:
                logger.warning("Audio samples are already collated; assuming they all have the same length")
            lengths = [audios.shape[-1]] * audios.shape[0]
            return audios, lengths

        elif isinstance(audios, Sequence) and len(audios) > 0 and isinstance(audios[0], torch.Tensor):
            if not torch.is_floating_point(audios[0]):
                raise ValueError("Invalid audio provided. Audio should be a floating point tensor")

            lengths = [audio.shape[-1] for audio in audios]
            audios = [audio.squeeze(0) if audio.ndim > 1 else audio for audio in audios]
            audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True, padding_value=0.0)
            return audios, lengths

        raise TypeError("Invalid audio provided. Audio should be one or more torch tensors or numpy arrays")


class GraniteSpeechProcessor:
    """Combines audio feature extraction with text tokenization for multimodal inputs."""

    def __init__(
        self,
        audio_processor: GraniteSpeechFeatureExtractor,
        tokenizer: Any,
        audio_token: str = "<|audio|>",
        **kwargs,
    ):
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer

        self.audio_token = (
            tokenizer.audio_token
            if hasattr(tokenizer, "audio_token")
            else audio_token
        )

    def __call__(
        self,
        text: Union[str, list[str]],
        audio: Optional[Union[torch.Tensor, list[torch.Tensor], np.ndarray, list[np.ndarray]]] = None,
        device: str = "cpu",
        **kwargs,
    ) -> dict:
        text = self._get_validated_text(text)
        prompt_strings = text

        if audio is not None:
            audio_inputs = self.audio_processor(audio, device=device)
            audio_embed_sizes = audio_inputs.pop("audio_embed_sizes")
            prompt_strings = self._expand_audio_tokens(text, audio_embed_sizes)
        else:
            audio_inputs = {}

        if "padding" not in kwargs:
            kwargs["padding"] = True
        text_inputs = self.tokenizer(prompt_strings, **kwargs)

        return {**text_inputs, **audio_inputs}

    def _expand_audio_tokens(
        self,
        text: list[str],
        audio_embed_sizes: Sequence[int]
    ) -> list[str]:
        # Expand single audio token to multiple tokens matching projected feature count
        prompt_strings = []
        num_replaced = 0

        for sample in text:
            while self.audio_token in sample:
                # Replace each audio token with N placeholders (N = projected feature count)
                sample = sample.replace(
                    self.audio_token,
                    "<placeholder>" * audio_embed_sizes[num_replaced],
                    1,
                )
                num_replaced += 1
            prompt_strings.append(sample)

        # Convert placeholders back to audio tokens for tokenization
        prompt_strings = [s.replace("<placeholder>", self.audio_token) for s in prompt_strings]

        return prompt_strings

    def _get_validated_text(
        self,
        text: Union[str, list[str]]
    ) -> list[str]:
        if isinstance(text, str):
            return [text]
        elif isinstance(text, list) and len(text) > 0 and isinstance(text[0], str):
            return text
        raise TypeError("Invalid text provided! Text should be a string or list of strings.")


__all__ = [
    "GraniteSpeech",
    "GraniteSpeechConfig",
    "GraniteSpeechFeatureExtractor",
    "GraniteSpeechProcessor",
    "is_peft_available",
]