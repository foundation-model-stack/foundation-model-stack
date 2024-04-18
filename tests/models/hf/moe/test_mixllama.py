import logging, traceback
import torch

from fms.models.hf.moe.mixllama import MixLlamaConfig, MixLlamaForCausalLM

logging.basicConfig(level=logging.INFO)

dtype = torch.float16
simple_arch = dict(
    hidden_size=128,
    intermediate_size=1024,
    num_hidden_layers=4,
    num_attention_heads=8,
    torch_dtype=dtype
)

configs = [
    # (name, cpu_ok, config)
    ("Basic LLaMa (no mixture)", True, 
        MixLlamaConfig(
            moe_mlp=False,
            output_router_logits=False,
            pretraining_tp=False,
            **simple_arch
        )),
    ("MixLLaMa with MLP MOE", True,
        MixLlamaConfig(
            num_local_experts=4,
            output_router_logits=True,
            pretraining_tp=False,
            **simple_arch
        )),
    ("MixLLaMa with MLP MOE always-on expert", True,
        MixLlamaConfig(
            num_experts_per_tok=2,
            num_local_experts=4,
            always_on_idx=0,
            output_router_logits=True,
            pretraining_tp=False,
            **simple_arch
        )),
    ("MixLLaMa with QUERY MOE", False,
        MixLlamaConfig(
            num_experts_per_tok=2,
            num_local_experts=4,
            moe_query=True,
            output_router_logits=True,
            pretraining_tp=False,
            _attn_implementation="flash_attention_2",
            **simple_arch
        )),
    ("MixLLaMa with ALL MOE always-on expert", False,
        MixLlamaConfig(
            num_experts_per_tok=2,
            num_local_experts=4,
            moe_query=True,
            moe_key=True,
            moe_value=True,
            output_router_logits=True,
            pretraining_tp=False,
            _attn_implementation="flash_attention_2",
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

    for idx, (name, cpu_ok, config) in enumerate(configs):
        
        try :

            logging.info("### TEST [{}] {}".format(idx, name))
            if not torch.cuda.is_available() and not cpu_ok:
                results.append("This test requires GPU")
                logging.info("This test requires GPU, skipping...")
                continue
            m = MixLlamaForCausalLM(config).to(dtype).to(device)
            o = m(dummy)
            o = m.generate(dummy)
            logging.info("### TEST [{}] Passed!".format(idx))
            del m
            results.append("Pass")
    
        except Exception as e:

            logging.info("### TEST [{}] Failed!".format(idx))
            traceback.print_exc()
            results.append("Fail")

    print("\n\n\n")
    for idx, (name, cpu_ok, config) in enumerate(configs):
        logging.info("TEST [{}]\tName: {}\tStatus: {}".format(idx, name, results[idx]))