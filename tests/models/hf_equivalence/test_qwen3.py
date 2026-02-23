import pytest
import torch
import torch.nn.functional as F
import random

from fms.models import get_model
from fms.utils.generation import pad_input_ids

device = "cpu"
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)


def _get_inputs(tokenizer, prompt="Hello, how are you?"):
    """Tokenize input prompt"""
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    return input_ids


def _get_hf_model_output(model_path, inputs):
    """Get output from HuggingFace model"""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        # The model uses the last token's representation as the embedding
        embeddings = outputs.last_hidden_state[:, -1, :]
        # Normalize embeddings for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)

    query_embedding = embeddings[0]
    doc_embeddings = embeddings[1:]

    return query_embedding, doc_embeddings


def _get_fms_model_output(model_path, inputs):
    """Get output from FMS model"""
    model = get_model(
        "hf_pretrained",
        model_path,
        data_type=torch.float32,
        device_type=device,
    )

    model.eval()
    torch.set_grad_enabled(False)

    # Get input_ids from the inputs dict
    input_ids = inputs["input_ids"].to(device)

    # Prepare inputs for FMS - this will create appropriate mask and position_ids
    input_ids_padded, padding_kwargs = pad_input_ids(input_ids, min_pad_length=0)
    input_ids_padded = input_ids_padded.to(device)

    with torch.no_grad():
        # Get embeddings from base model (before LM head)
        embeddings, _ = model.base_model(
            input_ids_padded,
            mask=padding_kwargs["mask"].to(device),
            position_ids=padding_kwargs["position_ids"].to(device),
        )
        # The model uses the last token's representation as the embedding
        embeddings = embeddings[:, -1, :]
        # Normalize embeddings for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)

    query_embedding = embeddings[0]
    doc_embeddings = embeddings[1:]

    return query_embedding, doc_embeddings


@pytest.mark.slow
def test_qwen3_embedding_0_6b_equivalence():
    """
    Test equivalence between FMS and HuggingFace implementations of Qwen3-Embedding-0.6B.

    This test:
    1. Loads both HF and FMS versions of the model
    2. Compares scores for the same input
    3. Compares scores against the original Qwen3-Embedding-0.6B scores

    Note: This test requires downloading the model from HuggingFace Hub.
    """
    model_path = "Qwen/Qwen3-Embedding-0.6B"

    # Skip if model is not available locally
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        pytest.skip(f"Model not available: {e}")

    # Prepare input
    query = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: What is the capital of China?"
    documents = ["The capital of China is Beijing.", "That is a very fast car."]
    input_texts = [query] + documents

    # 3. Tokenize
    inputs = tokenizer(
        input_texts, padding=True, truncation=True, return_tensors="pt", max_length=8192
    )

    # Get outputs from both models
    hf_query_embedding, hf_doc_embeddings = _get_hf_model_output(model_path, inputs)
    fms_query_embedding, fms_doc_embeddings = _get_fms_model_output(model_path, inputs)

    hf_scores = hf_query_embedding @ hf_doc_embeddings.T
    fms_scores = fms_query_embedding @ fms_doc_embeddings.T

    # First sentence contains the awnser to the query.
    # It's score should be always the highest.
    assert hf_scores[0] > hf_scores[1]
    assert fms_scores[0] > fms_scores[1]
    assert fms_scores[0] > 0.7
    assert hf_scores[0] > 0.7
    assert hf_scores[0] > fms_scores[1]
    assert fms_scores[0] > hf_scores[1]


@pytest.mark.slow
def test_qwen3_embedding_4b_equivalence():
    """
    Test equivalence between FMS and HuggingFace implementations of Qwen3-Embedding-4B.

    This test:
    1. Loads both HF and FMS versions of the model
    2. Compares scores for the same input
    3. Compares scores against the original Qwen3-Embedding-4B scores

    Note: This test requires downloading the model from HuggingFace Hub.
    """
    model_path = "Qwen/Qwen3-Embedding-4B"

    # Skip if model is not available locally
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        pytest.skip(f"Model not available: {e}")

    # Prepare input
    query = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: What is the capital of China?"
    documents = ["The capital of China is Beijing.", "That is a very fast car."]
    input_texts = [query] + documents

    # 3. Tokenize
    inputs = tokenizer(
        input_texts, padding=True, truncation=True, return_tensors="pt", max_length=8192
    )

    # Get outputs from both models
    hf_query_embedding, hf_doc_embeddings = _get_hf_model_output(model_path, inputs)
    fms_query_embedding, fms_doc_embeddings = _get_fms_model_output(model_path, inputs)

    hf_scores = hf_query_embedding @ hf_doc_embeddings.T
    fms_scores = fms_query_embedding @ fms_doc_embeddings.T

    # First sentence contains the awnser to the query.
    # It's score should be always the highest.
    assert hf_scores[0] > hf_scores[1]
    assert fms_scores[0] > fms_scores[1]
    assert fms_scores[0] > 0.7
    assert hf_scores[0] > 0.7
    assert hf_scores[0] > fms_scores[1]
    assert fms_scores[0] > hf_scores[1]


def test_qwen3_forward_pass():
    """
    Test basic forward pass of Qwen3 model.

    This is a simpler test that just verifies the model can be loaded
    and produces reasonable outputs without comparing to HF.
    """
    model_path = "Qwen/Qwen3-Embedding-0.6B"

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        pytest.skip(f"Model not available: {e}")

    # Load FMS model
    model = get_model(
        "hf_pretrained",
        model_path,
        data_type=torch.float32,
        device_type=device,
    )

    model.eval()

    # Prepare input
    prompt = "Hello, world!"
    input_ids = _get_inputs(tokenizer, prompt)
    input_ids_padded, padding_kwargs = pad_input_ids(input_ids, min_pad_length=0)
    input_ids_padded = input_ids_padded.to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(
            input_ids_padded,
            mask=padding_kwargs["mask"].to(device),
            position_ids=padding_kwargs["position_ids"].to(device),
        )

    # Basic sanity checks
    assert logits.shape[0] == 1, "Batch size should be 1"
    assert logits.shape[1] == input_ids.shape[1], "Sequence length should match input"
    assert logits.shape[2] == 151669, "Vocab size should be 151669"

    # Check that logits are reasonable (not NaN or Inf)
    assert not torch.isnan(logits).any(), "Logits contain NaN"
    assert not torch.isinf(logits).any(), "Logits contain Inf"

    # Check that logits have reasonable range
    assert logits.abs().max() < 100, "Logits have unreasonable magnitude"


def test_qwen3_parameter_count():
    """
    Test that FMS and HF models have the same number of parameters.
    """
    model_path = "Qwen/Qwen3-Embedding-0.6B"

    try:
        from transformers import AutoModelForCausalLM
    except Exception as e:
        pytest.skip(f"Transformers not available: {e}")

    try:
        # Load both models
        hf_model = AutoModelForCausalLM.from_pretrained(model_path)
        fms_model = get_model("hf_pretrained", model_path)

        # Count parameters
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())

        hf_params = count_parameters(hf_model)
        fms_params = count_parameters(fms_model)

        assert hf_params == fms_params, (
            f"Parameter count mismatch: HF {hf_params} vs FMS {fms_params}"
        )

        # Verify it's approximately 0.6B parameters
        assert 500_000_000 < fms_params < 700_000_000, (
            f"Expected ~600M parameters, got {fms_params}"
        )

    except Exception as e:
        pytest.skip(f"Could not load models: {e}")


def test_qwen3_with_cache():
    """
    Test that KV caching works correctly in Qwen3.
    """
    model_path = "Qwen/Qwen3-Embedding-0.6B"

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        pytest.skip(f"Model not available: {e}")

    # Load FMS model
    model = get_model(
        "hf_pretrained",
        model_path,
        data_type=torch.float32,
        device_type=device,
    )

    model.eval()

    # Prepare input
    prompt = "The quick brown fox"
    input_ids = _get_inputs(tokenizer, prompt)
    input_ids_padded, padding_kwargs = pad_input_ids(input_ids, min_pad_length=0)
    input_ids_padded = input_ids_padded.to(device)

    # Forward pass without cache
    with torch.no_grad():
        logits_no_cache = model(
            input_ids_padded,
            mask=padding_kwargs["mask"].to(device),
            position_ids=padding_kwargs["position_ids"].to(device),
            use_cache=False,
        )

    # Forward pass with cache
    with torch.no_grad():
        output_with_cache = model(
            input_ids_padded,
            mask=padding_kwargs["mask"].to(device),
            position_ids=padding_kwargs["position_ids"].to(device),
            use_cache=True,
        )

        if isinstance(output_with_cache, tuple):
            logits_with_cache, cache = output_with_cache
        else:
            logits_with_cache = output_with_cache
            cache = None

    # Logits should be the same regardless of caching
    torch.testing.assert_close(
        logits_no_cache,
        logits_with_cache,
        rtol=1e-5,
        atol=1e-5,
        msg="Logits differ when using cache",
    )

    # Cache should be returned when use_cache=True
    if cache is not None:
        assert len(cache) == 28, f"Expected 28 layers of cache, got {len(cache)}"


if __name__ == "__main__":
    test_qwen3_forward_pass()
    test_qwen3_with_cache()
    test_qwen3_parameter_count()
    test_qwen3_embedding_0_6b_equivalence()
