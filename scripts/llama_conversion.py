from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer
from fms.models.hf.llama.modeling_llama_hf import LLaMAHFForCausalLM

import fms.models.llama as llama

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
    