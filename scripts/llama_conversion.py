from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer
from fms.models.hf.llama.modeling_llama_hf import LLaMAHFForCausalLM

import fms.models.llama as llama

import argparse
import os

def convert_from_hf_to_fms(path_to_hf_model, path_to_fms_model):
    """Convert the HF checkpoints to FMS checkpoints to be used by this repo

    Args:
        path_to_hf_model (str): Path on disk to the HF checkpoint including the tokenizer
        path_to_fms_model (_type_): Path to save the FMS checkpoint including the tokenizer
    """
    # load the HF model and the tokenizer
    model = LLaMAHFForCausalLM.from_pretrained(path_to_hf_model)
    tokenizer = LlamaTokenizer.from_pretrained(path_to_hf_model)
    
    # convert to FMS HF
    fms_llama = llama.convert_hf_llama(model)
    fms_hf_llama = LLaMAHFForCausalLM.from_fms_model(fms_llama)
    
    # save checkpoint
    fms_hf_llama.save_pretrained(path_to_fms_model)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of HF LLaMA weights, which contains the model weights and the tokenizer",
        required=True
    )
    
    parser.add_argument(
        "--output_dir",
        help="Target location for FMS LLaMa weights"
        required=True
    )
    
    args = parser.parse_args()
    convert_from_hf_to_fms(args.input_dir, args.output_dir)
    
if __name__ == "__main__":
    main()
