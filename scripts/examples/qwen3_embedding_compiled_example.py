#!/usr/bin/env python3
"""
Example script to run Qwen3-Embedding-0.6B with torch.compile for optimized inference.

This script demonstrates:
1. Loading the Qwen3-Embedding-0.6B model from HuggingFace
2. Compiling the model with torch.compile for better performance
3. Computing embeddings for queries and documents
4. Calculating similarity scores

Usage:
    python scripts/qwen3_embedding_compiled_example.py
"""

import torch
import torch.nn.functional as F
from fms.models import get_model
from fms.utils.generation import pad_input_ids

# Configuration
MODEL_PATH = "Qwen/Qwen3-Embedding-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPILE_MODE = "default"  # Options: "default", "reduce-overhead", "max-autotune"


def load_and_compile_model(model_path: str, device: str, compile_mode: str = "default"):
    """
    Load the Qwen3 model and compile it for optimized inference.

    Args:
        model_path: Path to the model (HuggingFace model ID or local path)
        device: Device to run on ("cuda" or "cpu")
        compile_mode: Compilation mode for torch.compile

    Returns:
        Compiled model ready for inference
    """
    print(f"Loading model from {model_path}...")
    model = get_model(
        "hf_pretrained",
        model_path,
        data_type=torch.float32,
        device_type=device,
    )

    model.eval()
    torch.set_grad_enabled(False)

    print(f"Compiling model with mode='{compile_mode}'...")
    # Compile the base model for better performance
    # We compile base_model instead of the full model since we only need embeddings
    model = torch.compile(  # type: ignore[assignment,arg-type]
        model,  # type: ignore[arg-type]
        mode=compile_mode,
    )

    print("Model loaded and compiled successfully!")
    return model


def get_embeddings(model, input_ids, device):
    """
    Get normalized embeddings from the model.

    Args:
        model: The Qwen3 model
        input_ids: Input token IDs [batch_size, seq_len]
        device: Device to run on

    Returns:
        Normalized embeddings [batch_size, emb_dim]
    """
    # Prepare inputs for FMS - this will create appropriate mask and position_ids
    input_ids_padded, padding_kwargs = pad_input_ids(input_ids, min_pad_length=0)
    input_ids_padded = input_ids_padded.to(device)

    with torch.no_grad():
        # Get embeddings from base model (before LM head)
        embeddings = model(
            input_ids_padded,
            mask=padding_kwargs["mask"].to(device),
            position_ids=padding_kwargs["position_ids"].to(device),
        )
        # The model uses the last token's representation as the embedding
        embeddings = embeddings[:, -1, :]
        # Normalize embeddings for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


def compute_similarity_scores(
    query_embedding: torch.Tensor, doc_embeddings: torch.Tensor
):
    """
    Compute cosine similarity scores between query and documents.

    Args:
        query_embedding: Query embedding [emb_dim]
        doc_embeddings: Document embeddings [num_docs, emb_dim]

    Returns:
        Similarity scores [num_docs]
    """
    scores = query_embedding @ doc_embeddings.T
    return scores


def main():
    """Main execution function."""
    print("=" * 80)
    print("Qwen3-Embedding-0.6B Compiled Model Example")
    print("=" * 80)
    print()

    # Check if transformers is available
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(
            "Error: transformers library is required. Install with: pip install transformers"
        )
        return

    # Load tokenizer
    print(f"Loading tokenizer from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Make sure you have internet connection or the model is cached locally.")
        return

    # Load and compile model
    try:
        model = load_and_compile_model(MODEL_PATH, DEVICE, COMPILE_MODE)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print()
    print("-" * 80)
    print("Running inference example...")
    print("-" * 80)
    print()

    # Prepare example data
    query = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: What is the capital of China?"
    documents = [
        "The capital of China is Beijing.",
        "That is a very fast car.",
        "Beijing is a major city in China and serves as the nation's capital.",
    ]

    print(f"Query: {query}")
    print()
    print("Documents:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")
    print()

    # Tokenize inputs
    input_texts = [query] + documents
    inputs = tokenizer(
        input_texts, padding=True, truncation=True, return_tensors="pt", max_length=8192
    )
    input_ids = inputs["input_ids"].to(DEVICE)

    # Get embeddings
    print("Computing embeddings...")
    embeddings = get_embeddings(model, input_ids, DEVICE)

    # Split into query and document embeddings
    query_embedding = embeddings[0]
    doc_embeddings = embeddings[1:]

    # Compute similarity scores
    scores = compute_similarity_scores(query_embedding, doc_embeddings)

    # Display results
    print()
    print("-" * 80)
    print("Results:")
    print("-" * 80)
    print()
    print("Similarity Scores:")
    for i, (doc, score) in enumerate(zip(documents, scores), 1):
        print(f"  Document {i}: {score.item():.4f}")
        print(f'    "{doc}"')
        print()

    # Find most relevant document
    best_idx = int(scores.argmax().item())
    print(
        f"Most relevant document: Document {best_idx + 1} (score: {scores[best_idx].item():.4f})"
    )
    print(f'  "{documents[best_idx]}"')


if __name__ == "__main__":
    main()

# Made with Bob
