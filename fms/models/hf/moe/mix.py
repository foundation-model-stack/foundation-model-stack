import logging, shutil
from typing import List
from tqdm import tqdm
import gc

import torch
from transformers import (
    set_seed,
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    MistralForCausalLM,
    MixtralForCausalLM,
    MixtralConfig
)
from datasets import load_from_disk

from fms.models.hf.moe.mixllama import MixLlamaConfig, MixLlamaForCausalLM

MODELS_MOE_CLASS = {
    LlamaForCausalLM: (MixLlamaForCausalLM, MixLlamaConfig),
    MistralForCausalLM: (MixtralForCausalLM, MixtralConfig),
}

ROUTERS_NAME = {
    MixLlamaForCausalLM: ".router.",
    MixtralForCausalLM: ".gate."
}

EXPERTS_KEYS_MAPPING = {
    MixLlamaForCausalLM: {
        "mlp": "mlp.experts",
        "q_proj": "q_proj"
    },
    MixtralForCausalLM: {
        "mlp": "block_sparse_moe.experts",
    }
}

# dont know why mixtral change naming
KEYS_REMAP = {
    MixtralForCausalLM: {
        "mlp":{
            "w1":"gate_proj",
            "w3":"up_proj",
            "w2":"down_proj"
        }
    }
}

@torch.no_grad()
def mix(
    base_model:str,
    ingredients:List[str],
    modules_to_mix:List[str],
    positive_tokens:List[str]=[],
    num_samples:int=1000,
    num_experts_per_tok:int=2,
):
    
    set_seed(1399)

    logging.info("loading base model...")
    config_base = AutoConfig.from_pretrained(base_model)
    model_base = AutoModelForCausalLM.from_pretrained(base_model)
    model_base_type = type(model_base)
    MOE_MODEL_CLS, MOE_CFG_CLS = MODELS_MOE_CLASS[model_base_type]

    # SUPPORT CHECK
    assert num_experts_per_tok <= len(ingredients)

    assert len(modules_to_mix)>0 and "mlp" in modules_to_mix, \
        "Modules to mix must have at least 'mlp'!"
    
    assert any([isinstance(model_base, supported_model) for supported_model in MODELS_MOE_CLASS.keys()]), \
        "Model not supported! Only supports {}!".format(MODELS_MOE_CLASS.keys())
    
    KEYS_MAPPING = EXPERTS_KEYS_MAPPING[MOE_MODEL_CLS]
    assert all([mod in KEYS_MAPPING.keys() for mod in modules_to_mix]), \
        "Only supports {} for {}!".format(KEYS_MAPPING.keys(), model_base_type)
    
    if positive_tokens:
        assert len(positive_tokens) == len(ingredients)

    # /SUPPORT CHECK

    
    mixture_of_attention_head = "q_proj" in modules_to_mix
    sd_base = model_base.state_dict()
    
    logging.info("creating base model...")
    if mixture_of_attention_head:
        #NOTE: supports flash attention only
        config_base.torch_dtype = torch.float16
        config = MOE_CFG_CLS(
            num_local_experts= len(ingredients),
            moe_query_head=True,
            num_experts_per_tok=num_experts_per_tok,
            attn_implementation="flash_attention_2",
            **config_base.to_dict()
        )
    else:
        config = MOE_CFG_CLS(
            num_local_experts= len(ingredients),
            num_experts_per_tok=num_experts_per_tok,
            **config_base.to_dict()
        )
    
    logging.info(config)
    logging.info("creating moe model...")
    moe_model = MOE_MODEL_CLS(config) 
    moe_sd = moe_model.state_dict()
    moe_model_type = type(moe_model)
    router_name = ROUTERS_NAME[moe_model_type]
    
    experts_keys = [] # to be replaced with ingredients weights later
    routers_keys = [] # to be replaced later, if positive tokens are provided
    base_keys = [] # no use currently

    stem_param = 0
    experts_param = 0
    routers_param = 0

    for key in moe_sd:

        has_key_in_modules_to_mix = any([KEYS_MAPPING[x] in key for x in modules_to_mix])

        # stem
        if not has_key_in_modules_to_mix and not router_name in key:
            logging.info(f"copying {key} from base...")
            moe_sd[key].copy_(sd_base.pop(key))
            base_keys.append(key)
            stem_param += moe_sd[key].numel()

        # router
        elif router_name in key:
            if len(positive_tokens):
                logging.info(f"zeroing {key}...")
                moe_sd[key].zero_()
            else:
                logging.info(f"randomizing {key}...")
                moe_sd[key].random_()
            routers_keys.append(key)
            routers_param += moe_sd[key].numel()
        
        #  experts
        elif has_key_in_modules_to_mix:
            experts_keys.append(key)
            experts_param += moe_sd[key].numel()

        else:
            raise Exception("Something wrong!")
    
    del model_base
    del sd_base
    gc.collect()
    
    ## loading each ingredient models and and copy the weights to respectivce experts
    # all `experts_keys` should be overwritten with weightsafter this loop
    for expert_idx, path in enumerate(ingredients):

        logging.info("loading expert {} from {}...".format(expert_idx, path))
        ingred_model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=moe_model.config.torch_dtype
        )
        ingred_sd = ingred_model.state_dict()

        # for each specified module
        for module_name in modules_to_mix:

            keyword = f"{KEYS_MAPPING[module_name]}.{expert_idx}"
            matched_keys = [x for x in experts_keys if keyword in x]
            assert matched_keys, keyword

            # for each matched experts weight
            for key in matched_keys:
                key_cand = key.replace(keyword, module_name)

                # remap if necessary, for example mistral -> mixtral, [up_proj, down_proj, out_proj] -> [w1,w2,w3]
                if moe_model_type in KEYS_REMAP and module_name in KEYS_REMAP[moe_model_type]:
                    remap_dict = KEYS_REMAP[moe_model_type][module_name]
                    for _source_key, _target_key in remap_dict.items():
                        if _source_key in key_cand:
                            logging.info("remap {} to {}".format(_source_key, _target_key))
                            key_cand = key_cand.replace(_source_key, _target_key)

                logging.info("copying {} from expert {} to MOE {}...".format(key_cand, expert_idx, key))
                moe_sd[key].copy_(ingred_sd[key_cand])

                # for record
                experts_keys.remove(key)

        # replace the randomized weights if positive tokens are given
        if positive_tokens:

            with torch.no_grad():
                
                tokens_path = positive_tokens[expert_idx]
                tokens = load_from_disk(tokens_path)
                if isinstance(tokens, dict): tokens = tokens["train"]
                tokens = tokens["input_ids"][:num_samples]
                ingred_model.cuda().eval()
                logging.info("Computing hidden states using positive tokens from {}".format(tokens_path))
                for token_idx in tqdm(range(len(tokens))):
                    _hidden_states: List = ingred_model(
                        torch.tensor(tokens[token_idx]).unsqueeze(0).cuda(),
                        output_hidden_states=True,
                        return_dict=True
                    ).hidden_states[:-1]
                    _hidden_states = torch.stack(_hidden_states, dim=0).mean(-2) # average across sequence
                    hidden_states = _hidden_states.clone() if not token_idx else hidden_states + _hidden_states
                hidden_states = hidden_states.mean(1) # average across batch

                # for each specified module
                for module_name in modules_to_mix:

                    keyword = f"{KEYS_MAPPING[module_name]}"
                    if "." in keyword: keyword = keyword[:keyword.find(".")]
                    keyword += router_name
                    matched_keys = [x for x in routers_keys if keyword in x]
                    #NOTE: assume `routers_keys` are layer ordered

                    for layer_idx, key in enumerate(matched_keys):

                        logging.info("Replacing {}[{}] using hidden states computed.".format(key, expert_idx))
                        router_weight =  hidden_states[layer_idx]
                        router_weight /= router_weight.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                        moe_sd[key][expert_idx] += router_weight.cpu()

                        # for record
                        if expert_idx == len(ingredients)-1:
                            routers_keys.remove(key)

        del ingred_model
        del ingred_sd
        gc.collect()

    # END CHECK
    # ensure no weights are left empty/uncopied
    assert len(experts_keys) == 0, "Cannot match {}".format(experts_keys)

    if len(positive_tokens): assert len(routers_keys) == 0, "Cannot match {}".format(routers_keys)
    # /END CHECK

    # parameters
    logging.info("Stem parameters: {}".format(stem_param))
    logging.info("Experts parameters: {}".format(experts_param))
    logging.info("Routers parameters: {}".format(routers_param))
    logging.info("MOE total parameters (numel): {}".format(
        sum(p.numel() for p in moe_model.parameters())))
    logging.info("MOE total parameters : {}".format(stem_param + experts_param + routers_param))
    logging.info("MOE active parameters: {}".format(stem_param + routers_param + int(experts_param/len(ingredients)*num_experts_per_tok)))
    
    # wrapping up
    if isinstance(moe_model, MixLlamaForCausalLM):
        moe_model.config._name_or_path = "MixLlamaForCausalLM"
        moe_model.config.auto_map = {
            "AutoConfig": "configuration_mixllama.MixLlamaConfig",
            "AutoModel": "modeling_mixllama.MixLlamaModel",
            "AutoModelForCausalLM": "modeling_mixllama.MixLlamaForCausalLM"
        }
    
    return moe_model.to(torch.float16)


if __name__ == "__main__":

    import argparse, os
    from transformers import AutoTokenizer
    from fms.models.hf.moe.mixllama import modeling_mixllama, configuration_mixllama
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ingredients', nargs='+', required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--modules', nargs='+', default=["mlp"])
    parser.add_argument('--positive_tokens', nargs='+', default=[])
    args = parser.parse_args()
    
    model = mix(
        args.model_path,
        args.ingredients,
        args.modules,
        args.positive_tokens
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if isinstance(model, MixLlamaForCausalLM):
        shutil.copy(modeling_mixllama.__file__, os.path.join(args.output_dir, os.path.basename(modeling_mixllama.__file__)))
        shutil.copy(configuration_mixllama.__file__, os.path.join(args.output_dir, os.path.basename(configuration_mixllama.__file__)))