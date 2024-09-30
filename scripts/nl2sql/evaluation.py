"""
This module processes a JSONL file containing natural language utterances and SQL queries,
sends the utterances to an LLM to generate SQL, compares the generated SQL with the expected SQL
using Levenshtein distance, and outputs the results with scores.
"""

import json
import os
from typing import Any, Dict, List

import click
import Levenshtein
import requests
import torch
from dotenv import load_dotenv

from fms.models import get_model
from fms.utils import generation, tokenizers
from fms.utils.generation import generate


# Load environment variables from .env file
load_dotenv()

SQL_START = "SELECT"
SQL_END = "```"

SCORE_DIGITS = 2

PROMPT = "Question:\n{utterance}\n\nAnswer:\n"

torch.set_default_dtype(torch.bfloat16)
model = get_model(
    architecture="llama",
    variant="granite.code-8b",
    device_type="cuda",
    tie_heads=False,
    model_path=os.getenv("MODEL_FILE"),
    data_type=torch.bfloat16,
)
model.eval()
model.compile()
tokenizer = tokenizers.get_tokenizer("/gpfs/users/aviros/models/granite-8b-sql")


def send_llm_request(utterance: str, llm_url: str) -> str:
    """Send an utterance to the LLM and return the generated SQL."""
    payload = {
        "inputs": utterance,
        "parameters": {
            "best_of": 1,
            "decoder_input_details": False,
            "details": False,
            "do_sample": False,
            "max_new_tokens": 256,
            "repetition_penalty": 1.0,
            "return_full_text": False,
            "seed": None,
            "stop": [],
            "temperature": 0.01,
            "top_k": 50,
            "top_n_tokens": None,
            "top_p": 0.95,
            "truncate": None,
            "typical_p": 0.95,
            "watermark": True,
            "max_is_token_limit": False,
            "min_new_tokens": 2,
            "truncate_input_tokens": 0,
        },
    }
    # response = requests.post(
    #     llm_url, json=payload, headers={"Content-Type": "application/json"}
    # )
    # if not response.status_code == 200:
    #     return ""

    tokens = tokenizer.tokenize(utterance)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = [tokenizer.bos_token_id] + ids
    ids = torch.tensor(ids, dtype=torch.long, device="cuda")

    with torch.no_grad():
        result = generate(
            model,
            ids,
            max_new_tokens=256,
            use_cache=True,
            do_sample=False,
            max_seq_len=512,
        )

    print(
        f"Max/current reserved mem {torch.cuda.max_memory_reserved()/1024./1024/1024:.3f}G/{torch.cuda.memory_reserved()/1024./1024/1024:.3f}G"
    )
    print(
        f"Max/current allocated mem {torch.cuda.max_memory_allocated()/1024./1024/1024:.3f}G/{torch.cuda.memory_allocated()/1024./1024/1024:.3f}G"
    )
    torch.cuda.empty_cache()

    result = generation.trim_prefix(result, tokenizer.bos_token_id)
    result = generation.truncate_after_eos(result, tokenizer.eos_token_id)
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(result))


def extract_sql_from_response(response: str) -> str:
    """Extract the first SQL block from the LLM response."""
    sql_start = response.find(SQL_START)
    if sql_start == -1:
        return ""

    sql_end = response.find(SQL_END, sql_start)
    if sql_end == -1:
        return ""

    return response[sql_start:sql_end].strip()


def compare_sql(generated_query: str, expected_query: str) -> float:
    """Compare generated SQL and expected SQL using Levenshtein distance."""
    generated_query_cleaned = generated_query.replace("\n", "").strip().lower()
    expected_query_cleaned = expected_query.replace("```sql\n", "").replace("```", "")
    expected_query_cleaned = expected_query_cleaned.replace("\n", "").strip().lower()

    if generated_query_cleaned == expected_query_cleaned:
        return 1.0

    # Calculate Levenshtein distance ratio (normalized to a score between 0 and 1)
    lev_distance = Levenshtein.distance(generated_query_cleaned, expected_query_cleaned)
    max_len = max(len(generated_query_cleaned), len(expected_query_cleaned))
    score = 1 - (lev_distance / max_len)
    score = round(score, 4)

    return score


def process_line(
    line: Dict[str, Any], llm_url: str, score_distribution: List[float]
) -> Dict[str, Any]:
    """Process a single JSONL line, generate SQL, compare with expected, and add score."""

    # Extract the user question from 'messages'
    messages = line.get("messages", [])
    user_message = next(
        (msg["content"] for msg in messages if msg["role"] == "user"), ""
    )

    if not user_message:
        raise ValueError("No 'user' message found in the data.")

    assistant_msg = next((msg for msg in messages if msg["role"] == "assistant"), "")

    if not assistant_msg:
        raise ValueError("No 'assistant' response found in the data.")

    expected_query = assistant_msg["content"]  # type: ignore

    user_message = user_message.strip("\n")
    prompt = PROMPT.format(utterance=user_message)
    response = send_llm_request(prompt, llm_url)
    generated_query = extract_sql_from_response(response)
    score = compare_sql(generated_query, expected_query)

    print(f"Input prompt:\n{prompt}")
    print(f"Generated text:\n{generated_query}")
    print(f"Score: {score}\n")

    # Track the score for distribution calculation
    score_distribution.append(score)

    # Update the line with generated_query and score attributes
    assistant_msg["generated_query"] = generated_query  # type: ignore
    assistant_msg["score"] = score  # type: ignore

    return line


def get_score_distribution(score_distribution: List[float]) -> Dict[str, float]:
    """Returns the score distribution as a dictionary {range: percentage}."""
    total = len(score_distribution)
    counts = {
        "1.0": sum(1 for s in score_distribution if s == 1.0),
        "0.95-1.0": sum(1 for s in score_distribution if 0.95 <= s < 1.0),
        "0.90-0.95": sum(1 for s in score_distribution if 0.90 <= s < 0.95),
        "0.85-0.90": sum(1 for s in score_distribution if 0.85 <= s < 0.90),
        "0.80-0.85": sum(1 for s in score_distribution if 0.80 <= s < 0.85),
        "0.0-0.80": sum(1 for s in score_distribution if 0.0 < s < 0.80),
        "0.0": sum(1 for s in score_distribution if s == 0.0),
    }

    return {key: round((count / total) * 100, 2) for key, count in counts.items()}


@click.command()
@click.option("--input_file", default=None, help="Path to input JSONL file.")
@click.option("--output_file", default=None, help="Path to output JSONL file.")
@click.option("--statistics_file", default=None, help="Path to statistics JSON file.")
@click.option("--llm_url", default="bah", help="URL for the LLM endpoint.")
@click.option(
    "--num_samples",
    default=None,
    help="Number of samples to process from the input file.",
)
def main(
    input_file: str,
    output_file: str,
    statistics_file: str,
    llm_url: str,
    num_samples: int,
) -> None:
    """Main function to handle command-line arguments and process the inputfile."""
    # Load values from .env file if not provided by command line
    input_file = input_file or os.getenv("INPUT_FILE")  # type: ignore
    output_file = output_file or os.getenv("OUTPUT_FILE")  # type: ignore
    statistics_file = statistics_file or os.getenv("STATISTICS_FILE")  # type: ignore
    llm_url = llm_url or os.getenv("LLM_URL")  # type: ignore
    num_samples = int(num_samples or os.getenv("NUM_SAMPLES"))  # type: ignore

    if not input_file or not output_file or not llm_url:
        raise ValueError(
            "INPUT_FILE, OUTPUT_FILE, STATISTICS_FILE, and LLM_URL must be provided "
            "either as command line arguments or in the .env file."
        )

    score_distribution: List[float] = []

    counter = 0
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            line_data = json.loads(line)
            processed_data = process_line(line_data, llm_url, score_distribution)
            outfile.write(json.dumps(processed_data) + "\n")
            counter += 1

            if num_samples and counter == num_samples:
                break

        distribution_summary = get_score_distribution(score_distribution)

        # Print the score distribution summary
        for key, percentage in distribution_summary.items():
            print(f"{key}: {percentage:.2f}%")

        # Write the score distribution summary to the separate statistics file
        with open(statistics_file, "w") as stats_file:
            json.dump(
                {"score_distribution": distribution_summary}, stats_file, indent=2
            )


if __name__ == "__main__":
    main()
