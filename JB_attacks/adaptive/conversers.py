from ast import Not
import torch
import os
from typing import List
from language_models import GPT, HuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import VICUNA_PATH, LLAMA_7B_PATH, LLAMA_13B_PATH, LLAMA_70B_PATH, LLAMA3_8B_PATH, LLAMA3_70B_PATH, GEMMA_2B_PATH, GEMMA_7B_PATH, MISTRAL_7B_PATH, MIXTRAL_7B_PATH, R2D2_PATH, PHI3_MINI_PATH, TARGET_TEMP, TARGET_TOP_P   
import gc
from utils import extract_logprob

def load_target_model(target_model, target_system_message, target_token):
    targetLM = TargetLM(model_name = target_model, 
                        temperature = TARGET_TEMP, # init to 0
                        top_p = TARGET_TOP_P, # init to 1
                        system_message = target_system_message,
                        target_token = target_token
                        )
    return targetLM

class TargetLM():
    """
    Base class for target language models.
    
    Generates responses for prompts using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(self, 
            model_name: str, 
            temperature: float,
            top_p: float,
            system_message: str,
            target_token):
        
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.model, self.template, self.tokenizer = load_indiv_model(model_name)
        self.n_input_tokens = 0
        self.n_output_tokens = 0
        self.n_input_chars = 0
        self.n_output_chars = 0
        self.system_message = system_message
        self.target_token = target_token
        
    def get_response(self, prompts_list: List[str], max_n_tokens=None, temperature=None) -> List[dict]:
        batchsize = len(prompts_list)
        full_prompts = []

        if ("llama" in self.model_name) or ("mistral" in self.model_name):
            # No need to create 'conv' objects in this case
            if self.system_message is None:
                inputs_batch = [
                    [{"role": "user", "content": prompt}]
                    for prompt in prompts_list
                ]
            else:
                inputs_batch = [
                    [
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt}
                    ]
                    for prompt in prompts_list
                ]
            full_prompts = self.tokenizer.apply_chat_template(
                inputs_batch, tokenize=False, add_generation_prompt=True
            )
            
        else:
            NotImplementedError("Only Llama and Mistral models are supported for now.")

        outputs = self.model.generate(
            full_prompts,
            max_n_tokens=max_n_tokens,
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p
        )
        
        self.n_input_tokens += sum(output['n_input_tokens'] for output in outputs)
        self.n_output_tokens += sum(output['n_output_tokens'] for output in outputs)
        self.n_input_chars += sum(len(full_prompt) for full_prompt in full_prompts)
        self.n_output_chars += len([len(output['text']) for output in outputs])
        return outputs
    
    def read_log_prob(self, prompt):
       
        if self.system_message is None:
            inputs_batch = [
                [{"role": "user", "content": prompt}]
            ]
        else:
            inputs_batch = [
                [
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt}
                ]
            ]
            
        full_prompts = self.tokenizer.apply_chat_template(
            inputs_batch, tokenize=False, add_generation_prompt=True
        )
            
        outputs = self.model.generate(
            full_prompts,
            max_n_tokens= 1,
            temperature=self.temperature,
            top_p=self.top_p
        )
        
        logprob_dict = outputs[0]['logprobs'][0]
        logprob = extract_logprob(logprob_dict, self.target_token)
        
        return logprob


def load_indiv_model(model_name, device=None):
    model_path, template = get_model_path_and_template(model_name)
    
    if 'gpt' in model_name or 'together' in model_name:
        lm = GPT(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map= "auto", 
        ).eval()
            
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast= True,
        )
        
        if 'llama2' in model_path.lower():
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = 'left'
        elif 'vicuna' in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
        elif 'mistral' in model_path.lower() or 'mixtral' in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif 'llama-3' in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
            tokenizer.padding_side = 'left'
        elif not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        
        tokenizer.padding_side = 'left'  
            
        lm = HuggingFace(model_name, model, tokenizer)
    
    return lm, template, tokenizer

def get_model_path_and_template(model_name):
    full_model_dict={
        "gpt-4-0125-preview":{
            "path":"gpt-4",
            "template":"gpt-4"
        },
        "gpt-4-1106-preview":{
            "path":"gpt-4",
            "template":"gpt-4"
        },
        "gpt-4":{
            "path":"gpt-4",
            "template":"gpt-4"
        },
        "gpt-3.5-turbo": {
            "path":"gpt-3.5-turbo",
            "template":"gpt-3.5-turbo"
        },
        "gpt-3.5-turbo-1106": {
            "path":"gpt-3.5-turbo",
            "template":"gpt-3.5-turbo"
        },
        "vicuna":{
            "path":VICUNA_PATH,
            "template":"vicuna_v1.1"
        },
        "llama2":{
            "path":LLAMA_7B_PATH,
            "template":"llama-2"
        },
        "llama2-7b":{
            "path":LLAMA_7B_PATH,
            "template":"llama-2"
        },
        "llama2-13b":{
            "path":LLAMA_13B_PATH,
            "template":"llama-2"
        },
        "llama2-70b":{
            "path":LLAMA_70B_PATH,
            "template":"llama-2"
        },
        "llama3-8b":{
            "path":LLAMA3_8B_PATH,
            "template":"llama-2"
        },
        "llama3-70b":{
            "path":LLAMA3_70B_PATH,
            "template":"llama-2"
        },
        "gemma-2b":{
            "path":GEMMA_2B_PATH,
            "template":"gemma"
        },
        "gemma-7b":{
            "path":GEMMA_7B_PATH,
            "template":"gemma"
        },
        "mistral-7b":{
            "path":MISTRAL_7B_PATH,
            "template":"mistral"
        },
        "mixtral-7b":{
            "path":MIXTRAL_7B_PATH,
            "template":"mistral"
        },
        "r2d2":{
            "path":R2D2_PATH,
            "template":"zephyr"
        },
        "phi3":{
            "path":PHI3_MINI_PATH,
            "template":"llama-2"  # not used
        },
        "claude-instant-1":{
            "path":"claude-instant-1",
            "template":"claude-instant-1"
        },
        "claude-2":{
            "path":"claude-2",
            "template":"claude-2"
        },
        "palm-2":{
            "path":"palm-2",
            "template":"palm-2"
        }
    }
    # template = full_model_dict[model_name]["template"] if model_name in full_model_dict else "gpt-4"
    assert model_name in full_model_dict, f"Model {model_name} not found in `full_model_dict` (available keys {full_model_dict.keys()})"
    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template


    
