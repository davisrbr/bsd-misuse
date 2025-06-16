import common
from .language_models import APILiteLLM
import os 
import asyncio, inspect
from .config import FASTCHAT_TEMPLATE_NAMES, Model, ATTACK_TEMP, ATTACK_TOP_P


def load_attack_model(model_name):
    # create attack model and target model
    attackLM = AttackLM(model_name = model_name, 
                        max_n_tokens = 1024, 
                        max_n_attack_attempts = 10, 
                        category = ":)",
                        evaluate_locally = False
                        )
    
    return attackLM

def load_indiv_model(model_name, local = False, use_jailbreakbench= False):
    if local:
        raise NotImplementedError
    else:
        lm = APILiteLLM(model_name)
        
    return lm


class AttackLM():
    """
        Base class for attacker language models.
        
        Generates attacks for conversations using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(self, 
                model_name: str, 
                max_n_tokens: int, 
                max_n_attack_attempts: int, 
                category: str,
                evaluate_locally: bool):
        
        self.model_name = Model(model_name)
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts

        self.temperature = ATTACK_TEMP
        self.top_p = ATTACK_TOP_P

        self.category = category
        self.evaluate_locally = evaluate_locally
        self.model = load_indiv_model(model_name, 
                                      local = evaluate_locally, 
                                      use_jailbreakbench=False # Cannot use JBB as attacker
                                      )
        self.initialize_output = self.model.use_open_source_model
        self.template = FASTCHAT_TEMPLATE_NAMES[self.model_name]

    def preprocess_conversation(self, convs_list: list, prompts_list: list[str]):
        # For open source models, we can seed the generation with proper JSON
        init_message = ""
        if self.initialize_output:
            
            # Initalize the attack model's generated output to match format
            if len(convs_list[0].messages) == 0:# If is first message, don't need improvement
                init_message = '{"improvement": "","prompt": "'
            else:
                init_message = '{"improvement": "'
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            if self.initialize_output:
                conv.append_message(conv.roles[1], init_message)
        openai_convs_list = [conv.to_openai_api_messages() for conv in convs_list]
        return openai_convs_list, init_message
        
    def _generate_attack(self, openai_conv_list: list[list[dict]], init_message: str):
        batchsize = len(openai_conv_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize
        new_adv_prompts = [None] * batchsize
        
        # Continuously generate outputs until all are valid or max_n_attack_attempts is reached
        for attempt in range(self.max_n_attack_attempts):
            # Subset conversations based on indices to regenerate
            convs_subset = [openai_conv_list[i] for i in indices_to_regenerate]
            # Generate outputs 
            
            outputs_list = self.model.batched_generate(convs_subset,
                                                        max_n_tokens = self.max_n_tokens,  
                                                        temperature = self.temperature,
                                                        top_p = self.top_p,
                                                        extra_eos_tokens=["}"]
                                                    )
            

            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                full_output = init_message + full_output + "}" # Add end brace since we terminate generation on end braces
                attack_dict, json_str = common.extract_json(full_output)
                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    new_adv_prompts[orig_index] = json_str
                else:
                    new_indices_to_regenerate.append(orig_index)

            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate
            # If all outputs are valid, break
            if not indices_to_regenerate:
                break

        if any([output is None for output in valid_outputs]):
            raise ValueError(f"Failed to generate valid output after {self.max_n_attack_attempts} attempts. Terminating.")
        return valid_outputs, new_adv_prompts


    def get_attack(self, convs_list, prompts_list):
        """
        Generates responses for a batch of conversations and prompts using a language model. 
        Only valid outputs in proper JSON format are returned. If an output isn't generated 
        successfully after max_n_attack_attempts, it's returned as None.
        
        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.
        
        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """
        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."
        
        # Convert conv_list to openai format and add the initial message
        processed_convs_list, init_message = self.preprocess_conversation(convs_list, prompts_list)
        valid_outputs, new_adv_prompts = self._generate_attack(processed_convs_list, init_message)

        for jailbreak_prompt, conv in zip(new_adv_prompts, convs_list):
            # For open source models, we can seed the generation with proper JSON and omit the post message
            # We add it back here
            if self.initialize_output:
                jailbreak_prompt += self.model.post_message
            conv.update_last_message(jailbreak_prompt)
        
        return valid_outputs
    
    
    
# class AttackLM():
#     """
#         Base class for attacker language models.
        
#         Generates attacks for conversations using a language model. The self.model attribute contains the underlying generation model.
#     """
#     def __init__(self, 
#                 model_name: str, 
#                 max_n_tokens: int, 
#                 max_n_attack_attempts: int, 
#                 category: str,
#                 evaluate_locally: bool):
        
#         self.model_name = Model(model_name)
#         self.max_n_tokens = max_n_tokens
#         self.max_n_attack_attempts = max_n_attack_attempts

#         self.temperature = ATTACK_TEMP
#         self.top_p = ATTACK_TOP_P

#         self.category = category
#         self.evaluate_locally = evaluate_locally
#         self.model = load_indiv_model(model_name, 
#                                       local = evaluate_locally, 
#                                       use_jailbreakbench=False # Cannot use JBB as attacker
#                                       )
#         self.initialize_output = self.model.use_open_source_model
#         self.template = FASTCHAT_TEMPLATE_NAMES[self.model_name]

#     def preprocess_conversation(self, convs_list: list, prompts_list: list[str]):
#         # For open source models, we can seed the generation with proper JSON
#         init_message = ""
#         if self.initialize_output:
            
#             # Initalize the attack model's generated output to match format
#             if len(convs_list[0].messages) == 0:# If is first message, don't need improvement
#                 init_message = '{"improvement": "","prompt": "'
#             else:
#                 init_message = '{"improvement": "'
#         for conv, prompt in zip(convs_list, prompts_list):
#             conv.append_message(conv.roles[0], prompt)
#             if self.initialize_output:
#                 conv.append_message(conv.roles[1], init_message)
#         openai_convs_list = [conv.to_openai_api_messages() for conv in convs_list]
#         return openai_convs_list, init_message
      
#     async def _call_model(self, convs_subset):
#         """
#         Call self.model.batched_generate *either* directly (if it is async)
#         *or* in a worker thread (if it is synchronous).  In both cases we
#         return only after the LLM has answered.
#         """
#         kwargs = dict(
#             max_n_tokens = self.max_n_tokens,
#             temperature  = self.temperature,
#             top_p        = self.top_p,
#             extra_eos_tokens=["}"],
#         )

#         if inspect.iscoroutinefunction(self.model.batched_generate):
#             # already async → just await it
#             return await self.model.batched_generate(convs_subset, **kwargs)

#         # synchronous → run in default thread‑pool
#         loop = asyncio.get_running_loop()
#         func = lambda: self.model.batched_generate(convs_subset, **kwargs)
#         return await loop.run_in_executor(None, func)
      
#     async def _generate_attack(
#         self,
#         openai_conv_list: list[list[dict]],
#         init_message: str,
#     ):
#         batchsize = len(openai_conv_list)
#         indices_to_regenerate = list(range(batchsize))
#         valid_outputs   = [None] * batchsize
#         new_adv_prompts = [None] * batchsize

#         for attempt in range(self.max_n_attack_attempts):

#             convs_subset = [openai_conv_list[i] for i in indices_to_regenerate]

#             # ──► ONE call to the underlying model ──────────────────────────
#             outputs_list = await self._call_model(convs_subset)
#             # ────────────────────────────────────────────────────────────────

#             new_indices_to_regenerate = []
#             for i, full_output in enumerate(outputs_list):
#                 orig_index  = indices_to_regenerate[i]
#                 full_output = init_message + full_output + "}"
#                 attack_dict, json_str = common.extract_json(full_output)
#                 if attack_dict:
#                     valid_outputs[orig_index]   = attack_dict
#                     new_adv_prompts[orig_index] = json_str
#                 else:
#                     new_indices_to_regenerate.append(orig_index)

#             indices_to_regenerate = new_indices_to_regenerate
#             if not indices_to_regenerate:          # everything valid
#                 break

#         if any(o is None for o in valid_outputs):
#             raise ValueError(
#                 f"Failed to generate after {self.max_n_attack_attempts} attempts."
#             )
#         return valid_outputs, new_adv_prompts


#     async def get_attack(self, convs_list, prompts_list):
#         assert len(convs_list) == len(prompts_list), "Mismatched batch sizes"

#         processed, init_msg = self.preprocess_conversation(convs_list, prompts_list)
#         valid, new_prompts  = await self._generate_attack(processed, init_msg)

#         for jailbreak_prompt, conv in zip(new_prompts, convs_list):
#             if self.initialize_output:
#                 jailbreak_prompt += self.model.post_message
#             conv.update_last_message(jailbreak_prompt)

#         return valid