from IPython import embed
import torch
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

MODEL_CLS = {
    "bge": AutoModel,
    "llama": AutoModelForCausalLM,
    "llama-lora": AutoModelForCausalLM,
    "qwen": AutoModelForCausalLM,
    "qwen-lora": AutoModelForCausalLM,
}



def load_model(model_args, for_eval=False):
    if model_args.model_dtype == 'fp16':
        model_dtype = torch.float16
    elif model_args.model_dtype == 'fp32':
        model_dtype = torch.float32
    elif model_args.model_dtype == 'bf16':
        model_dtype = torch.bfloat16
    else:
        model_dtype = 'auto'
        
    if model_args.model_name_or_path is None:
        return None, None
    
    # load model
    model_cls = MODEL_CLS[model_args.model_type]    
    if "lora" in model_args.model_type:
        if for_eval:
            peft_model_name = model_args.model_name_or_path
            config = PeftConfig.from_pretrained(peft_model_name)
            base_model = model_cls.from_pretrained(config.base_model_name_or_path, torch_dtype=model_dtype)
            model = PeftModel.from_pretrained(base_model, peft_model_name, config=config)
            model = model.merge_and_unload()
        else:
            model = model_cls.from_pretrained(model_args.model_name_or_path, torch_dtype=model_dtype)
            model.enable_input_require_grads() # add this when using gradient checkpointings
            peft_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type="CAUSAL_LM",
                    r=32,
                    lora_alpha=64,
                    lora_dropout=0.1,
                    target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
                    inference_mode=False
                )
            model = get_peft_model(model, peft_config)
    else:
        model = model_cls.from_pretrained(model_args.model_name_or_path, torch_dtype=model_dtype)
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    assert tokenizer.pad_token != tokenizer.eos_token
    
    if for_eval:
        model.eval()
    
    return model, tokenizer

def retriever_forward(model, 
                      model_type: str, 
                      inputs: dict,
                      normalize_emb: bool):
    
    if model_type in ['bge']:
        output = model(**inputs)
        embs = output.last_hidden_state[:, 0]
    else:
        assert KeyError("model_type not supported")
        
    if normalize_emb:
        embs = torch.nn.functional.normalize(embs, p=2, dim=-1)
        
    return embs