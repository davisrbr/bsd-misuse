from email import message
import os 
from fastapi import responses
import litellm
from .config import TOGETHER_MODEL_NAMES, LITELLM_MODEL_NAMES, LITELLM_TEMPLATES, API_KEY_NAMES, Model
import numpy as np
import time
from gray_swan import GraySwan
litellm.turn_off_message_logging=True
litellm.suppress_debug_info = True
litellm.set_verbose = False

class LanguageModel():
    def __init__(self, model_name):
        self.model_name = Model(model_name)
    
    def batched_generate(self, prompts_list: list, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError
    
class APILiteLLM(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "ERROR: API CALL FAILED."
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20

    def __init__(self, model_name):
        super().__init__(model_name)
        self.api_key = self.get_api_key(self.model_name)
        self.litellm_model_name = self.get_litellm_model_name(self.model_name)
        litellm.drop_params=True
        self.set_eos_tokens(self.model_name)
        
    def get_litellm_model_name(self, model_name):
        if model_name in TOGETHER_MODEL_NAMES:
            litellm_name = TOGETHER_MODEL_NAMES[model_name]
            self.use_open_source_model = True
        
        elif model_name in LITELLM_MODEL_NAMES:
            litellm_name = LITELLM_MODEL_NAMES[model_name]
            self.use_open_source_model = False
            
        else:
            if self.use_open_source_model:
                # Output warning, there should be a TogetherAI model name
                print(f"Warning: No TogetherAI model name for {model_name}.")
            litellm_name = model_name
        return litellm_name
    
    def set_eos_tokens(self, model_name):
        if "llama_3" in str(model_name):
            self.eos_tokens= ["<|eot_id|>"]
        elif "mistral" in str(model_name):
            self.eos_tokens= ["</s>", "[/INST]"]
        elif self.use_open_source_model:
            self.eos_tokens = LITELLM_TEMPLATES[model_name]["eos_tokens"]     
        else:
            self.eos_tokens = None

    def _update_prompt_template(self):
        # We manually add the post_message later if we want to seed the model response
        if self.model_name in LITELLM_TEMPLATES:
            litellm.register_prompt_template(
                initial_prompt_value=LITELLM_TEMPLATES[self.model_name]["initial_prompt_value"],
                model=self.litellm_model_name,
                roles=LITELLM_TEMPLATES[self.model_name]["roles"]
            )
            self.post_message = LITELLM_TEMPLATES[self.model_name]["post_message"]
        else:
            self.post_message = ""
        
    def get_api_key(self, model_name):
        self.use_open_source_model = False
        environ_var = API_KEY_NAMES[model_name]
        try:
            return os.environ[environ_var]  
        except KeyError:
            print(f"Missing API key, for {model_name}, please enter your API key by running: export {environ_var}='your-api-key-here'")
    
    
    def batched_generate(self, convs_list: list[list[dict]], 
                         max_n_tokens: int, 
                         temperature: float, 
                         top_p: float = 1,
                         extra_eos_tokens: list[str] = None, 
                         max_tries = 3) -> list[str]: 
        
        # if self.use_open_source_model:
        #     eos_tokens = self.eos_tokens.copy() 
            
        #     if extra_eos_tokens:
        #         eos_tokens.extend(extra_eos_tokens)
                
        # if self.use_open_source_model:
        #     self._update_prompt_template()
        
        # if eos_tokens:
        #     outputs = litellm.batch_completion(
        #         model=self.litellm_model_name, 
        #         messages=convs_list,
        #         temperature=temperature,
        #         top_p=top_p,
        #         max_tokens=max_n_tokens,
        #         num_retries=self.API_MAX_RETRY,
        #         seed=0,
        #         stop=eos_tokens,
        #     )
        # else:
        outputs = litellm.batch_completion(
            model=self.litellm_model_name, 
            messages=convs_list,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_n_tokens,
            num_retries=self.API_MAX_RETRY,
            seed=0,
        )
        
        responses = [output["choices"][0]["message"].content for output in outputs]
        
        return responses