import logging, traceback
import gc, shutil
import torch

from fms.models.hf.moe.mix import mix

from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM
)

TMP_DIR = "./m1"

dtype = torch.float16
simple_arch = dict(
    hidden_size=128,
    intermediate_size=1024,
    num_hidden_layers=4,
    num_attention_heads=8,
    torch_dtype=dtype
)

logging.basicConfig(level=logging.INFO)

test_cases = [
    # (name, cpu_ok, modules_to_mix, model_class, config)
    ("mistral mlp mix", True, ["mlp"],
        MistralForCausalLM, MistralConfig(
            **simple_arch
        )),
    ("Llama mlp mix", True, ["mlp"],
        LlamaForCausalLM, LlamaConfig(
            **simple_arch
        )),
    ("Llama mlp query mix", False, ["mlp", "q_proj"],
        LlamaForCausalLM, LlamaConfig(
            **simple_arch
        )),
    ("Llama mlp all mix", False, ["mlp", "q_proj", "k_proj", "v_proj"],
        LlamaForCausalLM, LlamaConfig(
            **simple_arch
        )),
]

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dummy = torch.ones((1,8)).int().to(device)
    results = []
    
    for idx, (name, cpu_ok, modules_to_mix, cls, config) in enumerate(test_cases):
        
        try :
            logging.info("### TEST [{}] {}".format(idx, cls))
            if not torch.cuda.is_available() and not cpu_ok:
                results.append("This test requires GPU")
                logging.info("This test requires GPU")
                continue
            m = cls(config)
            m.save_pretrained("m1")
            del m
            gc.collect()
            logging.info("mixing...")
            m = mix(TMP_DIR, [TMP_DIR]*4, modules_to_mix)
            del m
            m = AutoModelForCausalLM.from_pretrained(TMP_DIR, trust_remote_code=True)
            m.to(device)(dummy)
            del m
            logging.info("### TEST [{}] Passed!".format(idx))
            results.append("Pass")
        
        except Exception as e:
            
            logging.info("### TEST [{}] Failed!".format(idx))
            traceback.print_exc()
            results.append("Fail")

    shutil.rmtree(TMP_DIR)
    print("\n\n\n")
    for idx, (name, cpu_ok, modules_to_mix, cls, config) in enumerate(test_cases):
        logging.info("TEST [{}]\tName: {}\tStatus: {}".format(idx, name, results[idx]))