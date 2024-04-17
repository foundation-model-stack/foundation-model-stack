import logging, traceback
import gc, shutil
import torch

from fms.models.hf.moe import mix

from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM
)

TMP_DIR = "./m1"

hidden_size = 128
intermediate_size = 1024
num_hidden_layers = 4
num_heads = 4

logging.basicConfig(level=logging.INFO)

test_cases = [
    # (model_class, config_class, modules_to_mix, cpu_ok)
    (MistralForCausalLM, MistralConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
    ), ["mlp"], True),
    (LlamaForCausalLM, LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        output_router_logits=True,
        pretraining_tp=False
    ), ["mlp"], True),
    (LlamaForCausalLM, LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        output_router_logits=True,
        pretraining_tp=False
    ), ["mlp", "q_proj"], False),
]

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    try :
        for idx, (cls, config, modules_to_mix, cpu_ok) in enumerate(test_cases):
            logging.info("### TEST [{}] {}".format(idx, cls))
            if not torch.cuda.is_available() and not cpu_ok:
                logging.info("This test requires GPU")
                continue
            m = cls(config)
            m.save_pretrained("m1")
            del m
            gc.collect()
            logging.info("mixing...")
            m = mix.mix(TMP_DIR, [TMP_DIR]*4, modules_to_mix)
            m.save_pretrained(TMP_DIR)
            del m
            m = AutoModelForCausalLM.from_pretrained(TMP_DIR, trust_remote_code=True)
            del m
            logging.info("### TEST [{}] Passed!".format(idx))
    except Exception as e:
        logging.info("### TEST [{}] Failed!".format(idx))
        traceback.print_exc()

    shutil.rmtree("./m1")
    
