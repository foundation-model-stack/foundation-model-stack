import logging, traceback
import torch

from fms.models.hf.moe.mixllama import MixLlamaConfig, MixLlamaForCausalLM

logging.basicConfig(level=logging.INFO)

dtype = torch.float16

configs = [
    # (config, cpu_ok)
    ( MixLlamaConfig(
        hidden_size=128,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_experts_per_tok=2,
        num_local_experts=4,
        output_router_logits=True,
        pretraining_tp=False
    ), True),
    ( MixLlamaConfig(
        hidden_size=128,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_experts_per_tok=2,
        num_local_experts=4,
        output_router_logits=True,
        pretraining_tp=False,
        num_moa_experts=4,
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2"
    ), False),
]

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dummy = torch.ones((1,8)).int().to(device)

    try :
        for idx, (config, cpu_ok) in enumerate(configs):
            logging.info("### TEST [{}] {}".format(idx, type(config)))
            if not torch.cuda.is_available() and not cpu_ok:
                logging.info("This test requires GPU")
                continue
            m = MixLlamaForCausalLM(config).to(dtype).to(device)
            o = m(dummy)
            o = m.generate(dummy)
            logging.info("### TEST [{}] Passed!".format(idx))
            del m
    except Exception as e:
        logging.info("### TEST [{}] Failed!".format(idx))
        traceback.print_exc()