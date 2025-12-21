"""
HuggingFace Equivalence Tests for GraniteSpeech

This module tests that FMS GraniteSpeech implementation produces identical
outputs to the HuggingFace reference implementation.

Test Structure (following FMS equivalence test standards):
1. Signature comparison - compressed output distribution comparison
2. Generation comparison - actual transcription output comparison
"""

import pytest
import torch

from fms.models import get_model
from fms.testing.comparison import (
    ModelSignatureParams,
    HFModelSignatureParams,
    compare_model_signatures,
)

device = "cuda"
torch.set_default_dtype(torch.float32)


# =============================================================================
# Shared Utilities
# =============================================================================


def _get_audio_inputs(processor):
    """Get sample audio inputs from LibriSpeech dummy dataset."""
    from datasets import load_dataset

    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    sample = ds[0]
    audio = sample["audio"]["array"]
    text = ["Transcribe the following audio: <|audio|>"]

    inputs = processor(
        text=text,
        audio=audio,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def _load_hf_model(model_path):
    """Load HuggingFace model."""
    from transformers import GraniteSpeechForConditionalGeneration

    model = GraniteSpeechForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=device,
    )
    model.eval()
    return model


def _load_fms_model(model_path):
    """Load FMS model using hf_pretrained."""
    # FIXME: The ibm-granite/granite-speech-3.3-2b HuggingFace repo contains both
    # old 3-shard checkpoint files (*-of-00003.safetensors, output_dim=42) and new
    # 4-shard files (*-of-00004.safetensors, output_dim=256). FMS loads all safetensor
    # files via glob instead of using model.safetensors.index.json, causing shape
    # mismatches. Workaround: download with ignore_patterns to exclude old files.
    # See: https://huggingface.co/ibm-granite/granite-speech-3.3-2b
    # TODO: Remove this workaround once IBM cleans up the HF repo or FMS uses index.json
    if "granite-speech-3.3-2b" in model_path:
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(
            model_path,
            ignore_patterns=["*-of-00003.safetensors"],
        )

    model = get_model(
        "hf_pretrained",
        model_path,
        data_type=torch.float32,
        device_type=device,
    )
    model.eval()
    return model


# =============================================================================
# Option 1: Signature Comparison
# Uses compressed output distribution (max - min across vocabulary)
# Following the pattern from test_llava_next.py
# =============================================================================


def _get_fms_logits(model_output):
    """Extract logits from FMS model output (returns tuple)."""
    logits, _ = model_output
    return logits


def _get_hf_logits(model_output):
    """Extract logits from HF model output."""
    return model_output.logits


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Granite Speech HF equivalence test requires CUDA"
)
def test_granite_speech_2b_signature_equivalence():
    """
    Option 1: Signature comparison for GraniteSpeech 2B.

    Compares compressed output distribution (max - min range across vocabulary).
    This is more robust than raw logit comparison.
    """
    from transformers import GraniteSpeechProcessor

    model_path = "ibm-granite/granite-speech-3.3-2b"
    processor = GraniteSpeechProcessor.from_pretrained(model_path)
    inputs = _get_audio_inputs(processor)

    hf_model = _load_hf_model(model_path)
    fms_model = _load_fms_model(model_path)

    # Prepare signature params following LLaVA pattern:
    # - params: list of main input param names
    # - inp: the main input tensor (input_ids)
    # - other_params: additional multimodal inputs

    fms_signature_params = ModelSignatureParams(
        model=fms_model,
        params=["input_ids"],  # Main param name
        inp=inputs["input_ids"],  # Main input
        other_params={  # Additional multimodal inputs
            "input_features": inputs["input_features"],
            "input_features_mask": inputs.get("input_features_mask"),
            "attention_mask": inputs.get("attention_mask"),
        },
        logits_getter_fn=_get_fms_logits,
    )

    hf_signature_params = HFModelSignatureParams(
        model=hf_model,
        params=["input_ids"],
        inp=inputs["input_ids"],
        other_params={
            "input_features": inputs["input_features"],
            "input_features_mask": inputs.get("input_features_mask"),
            "attention_mask": inputs.get("attention_mask"),
            "return_dict": True,
        },
        logits_getter_fn=_get_hf_logits,
    )

    # Compare signatures (uses np.allclose with atol=1e-3, rtol=1e-5)
    compare_model_signatures(fms_signature_params, hf_signature_params)
    print("Signature comparison passed!")


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Granite Speech HF equivalence test requires CUDA"
)
def test_granite_speech_8b_signature_equivalence():
    """Option 1: Signature comparison for GraniteSpeech 8B."""
    from transformers import GraniteSpeechProcessor

    model_path = "ibm-granite/granite-speech-3.3-8b"
    processor = GraniteSpeechProcessor.from_pretrained(model_path)
    inputs = _get_audio_inputs(processor)

    hf_model = _load_hf_model(model_path)
    fms_model = _load_fms_model(model_path)

    fms_signature_params = ModelSignatureParams(
        model=fms_model,
        params=["input_ids"],
        inp=inputs["input_ids"],
        other_params={
            "input_features": inputs["input_features"],
            "input_features_mask": inputs.get("input_features_mask"),
            "attention_mask": inputs.get("attention_mask"),
        },
        logits_getter_fn=_get_fms_logits,
    )

    hf_signature_params = HFModelSignatureParams(
        model=hf_model,
        params=["input_ids"],
        inp=inputs["input_ids"],
        other_params={
            "input_features": inputs["input_features"],
            "input_features_mask": inputs.get("input_features_mask"),
            "attention_mask": inputs.get("attention_mask"),
            "return_dict": True,
        },
        logits_getter_fn=_get_hf_logits,
    )

    compare_model_signatures(fms_signature_params, hf_signature_params)
    print("Signature comparison passed!")


# =============================================================================
# Option 2: Generation Comparison
# Compares actual transcription output (following test_granite_vision.py pattern)
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Granite Speech HF equivalence test requires CUDA"
)
def test_granite_speech_2b_generation_equivalence():
    """
    Option 2: Generation comparison for GraniteSpeech 2B.

    Compares actual generated transcription output.
    This is the ultimate test - if transcriptions match, models are equivalent.
    """
    from transformers import GraniteSpeechProcessor

    model_path = "ibm-granite/granite-speech-3.3-2b"
    processor = GraniteSpeechProcessor.from_pretrained(model_path)
    inputs = _get_audio_inputs(processor)

    hf_model = _load_hf_model(model_path)

    # HF generation
    with torch.no_grad():
        hf_output = hf_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )
    hf_transcription = processor.batch_decode(hf_output, skip_special_tokens=True)[0]

    # FMS generation (using HF's generate since FMS model is wrapped)
    # Note: FMS GraniteSpeech currently uses HF's generation infrastructure
    fms_model = _load_fms_model(model_path)

    # For FMS, we need to use FMS's generation utility
    from fms.utils.generation import generate

    with torch.no_grad():
        # Prepare inputs for FMS generate
        input_ids = inputs["input_ids"]
        extra_kwargs = {
            "input_features": inputs["input_features"],
            "input_features_mask": inputs.get("input_features_mask"),
            "attention_mask": inputs.get("attention_mask"),
        }

        fms_output = generate(
            fms_model,
            input_ids,
            max_new_tokens=100,
            do_sample=False,
            use_cache=True,
            prepare_model_inputs_hook=fms_model.prepare_inputs_for_generation,
            extra_kwargs=extra_kwargs,
        )

    fms_transcription = processor.batch_decode(fms_output, skip_special_tokens=True)[0]

    print(f"HF transcription: {hf_transcription}")
    print(f"FMS transcription: {fms_transcription}")

    # Compare transcriptions
    assert hf_transcription == fms_transcription, (
        f"Transcription mismatch!\n"
        f"HF:  {hf_transcription}\n"
        f"FMS: {fms_transcription}"
    )
    print("Generation comparison passed!")


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Granite Speech HF equivalence test requires CUDA"
)
def test_granite_speech_8b_generation_equivalence():
    """Option 2: Generation comparison for GraniteSpeech 8B."""
    from transformers import GraniteSpeechProcessor

    model_path = "ibm-granite/granite-speech-3.3-8b"
    processor = GraniteSpeechProcessor.from_pretrained(model_path)
    inputs = _get_audio_inputs(processor)

    hf_model = _load_hf_model(model_path)

    with torch.no_grad():
        hf_output = hf_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )
    hf_transcription = processor.batch_decode(hf_output, skip_special_tokens=True)[0]

    fms_model = _load_fms_model(model_path)
    from fms.utils.generation import generate

    with torch.no_grad():
        input_ids = inputs["input_ids"]
        extra_kwargs = {
            "input_features": inputs["input_features"],
            "input_features_mask": inputs.get("input_features_mask"),
            "attention_mask": inputs.get("attention_mask"),
        }

        fms_output = generate(
            fms_model,
            input_ids,
            max_new_tokens=100,
            do_sample=False,
            use_cache=True,
            prepare_model_inputs_hook=fms_model.prepare_inputs_for_generation,
            extra_kwargs=extra_kwargs,
        )

    fms_transcription = processor.batch_decode(fms_output, skip_special_tokens=True)[0]

    print(f"HF transcription: {hf_transcription}")
    print(f"FMS transcription: {fms_transcription}")

    assert hf_transcription == fms_transcription, (
        f"Transcription mismatch!\n"
        f"HF:  {hf_transcription}\n"
        f"FMS: {fms_transcription}"
    )
    print("Generation comparison passed!")


# =============================================================================
# Combined Tests (for backward compatibility)
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Granite Speech HF equivalence test requires CUDA"
)
def test_granite_speech_3_3_2b_equivalence():
    """
    Full equivalence test for GraniteSpeech 2B.
    Runs both signature and generation comparisons.
    """
    test_granite_speech_2b_signature_equivalence()
    test_granite_speech_2b_generation_equivalence()


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Granite Speech HF equivalence test requires CUDA"
)
def test_granite_speech_3_3_8b_equivalence():
    """
    Full equivalence test for GraniteSpeech 8B.
    Runs both signature and generation comparisons.
    """
    test_granite_speech_8b_signature_equivalence()
    test_granite_speech_8b_generation_equivalence()


if __name__ == "__main__":
    test_granite_speech_3_3_2b_equivalence()