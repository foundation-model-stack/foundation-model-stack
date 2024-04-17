Rapid creation of MOE Model.

![mixing_routine](/static/mix_routine.png)

### Example: MixLLaMa
```cli
python3 fms/models/moe/mix.py \
    --output_dir path_to_save/my_moellama \
    --modules mlp q_proj \
    --model_path  TheBloke/Llama-2-7B-fp16 \
    --ingredients \
        AdaptLLM/law-chat \
        AdaptLLM/finance-chat \
        AdaptLLM/medicine-chat
```

### Example: Mixtral
```cli
python3 fms/models/moe/mix.py \
    --output_dir path_to_save/my_mixtral \
    --modules mlp \
    --model_path EmbeddedLLM/Mistral-7B-Merge-14-v0.1 \
    --ingredients \
        mistralai/Mistral-7B-Instruct-v0.2 \
        openchat/openchat-3.5-1210 \
        maywell/PiVoT-0.1-Starling-LM-RP \
        beowolx/CodeNinja-1.0-OpenChat-7B
```
