from text_generation_server.models import get_model
from text_generation_server.pb import generate_pb2
from typing import List
import time
import torch

template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"

text = template.format(
    "Provide a list of instructions for preparing chicken soup."
)

def __generate_prefill_request(id: int, batch_size: int, in_tokens: List[int], num_new_tokens: List[int]):

    out = generate_pb2.PrefillRequest(
        batch=generate_pb2.Batch(
            id=id,
            requests=[
                generate_pb2.Request(
                    id=i, inputs=text, input_length=in_tokens[i], truncate=True, max_output_length=num_new_tokens[i],
                    parameters=generate_pb2.NextTokenChooserParameters(
                        temperature=0.0,
                    )
                ) for i in range(batch_size)
            ]
        )
    )
    return out

model = get_model(
    model_name="/net/storage149/mnt/md0/jmrosenk/llama_weights/hf/7B-F",
    revision=None,
    deployment_framework="hf_custom_tp",
    dtype_str="float16",
    quantize=None,
    max_sequence_length=2048
)


input_lengths = [49]
num_new_tokens = [50]

request1 = __generate_prefill_request(0, 1, input_lengths, num_new_tokens)

batch1, errors = model.batch_type.from_pb(
    request1.batch,
    tokenizer=model.tokenizer,
    dtype=model.dtype,
    device=model.device,
    embeddings_lookup=model.word_embeddings,
    prefix_cache=model.prefix_cache,
    use_position_ids=model.use_position_ids,
)


model.generate_token(batch1, first=True, for_concat=False)

for i in range(max(num_new_tokens)-1):
    t0 = time.time_ns()
    model.generate_token(batch1)
    torch.cuda.synchronize(device=model.device)
    t_tok = time.time_ns()-t0
    print("t_tok: %.2f ms" % (t_tok/1000.0/1000.0))