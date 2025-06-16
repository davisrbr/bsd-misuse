from enum import Enum
ATTACK_TEMP = 1
ATTACK_TOP_P = 0.9



## MODEL PARAMETERS ##
class Model(Enum):
    vicuna = "vicuna"
    llama_2 = "llama-2"
    llama_3= "llama-3"
    gpt_3_5 = "gpt-3.5-turbo-1106"
    gpt_4 = "gpt-4o"
    claude_1 = "claude-instant-1.2"
    claude_2 = "claude-2.1"
    gemini = "gemini-pro"
    mixtral = "mixtral"
    mistral = "mistral"
    claude_3 = "claude-3"
    Deepseek = "Deepseek"
    o1 = "o1"
    qwen = "qwen"

MODEL_NAMES = [model.value for model in Model]


HF_MODEL_NAMES: dict[Model, str] = {
    Model.llama_2: "meta-llama/Llama-2-7b-chat-hf",
    Model.llama_3: "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    Model.vicuna: "lmsys/vicuna-13b-v1.5",
}

TOGETHER_MODEL_NAMES: dict[Model, str] = {
    Model.llama_2: "together_ai/meta-llama/Llama-2-7b-chat-hf",
    Model.llama_3: "together_ai/meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    Model.vicuna: "together_ai/smahdi/lmsys/vicuna-13b-v1.5-1e4b3826",
}

LITELLM_MODEL_NAMES: dict[Model, str] = {
    Model.mixtral: "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
    Model.qwen: "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
    Model.Deepseek: "together_ai/deepseek-ai/DeepSeek-R1",
    Model.gpt_4: "openai/gpt-4o",
    Model.gpt_3_5: "openai/gpt-3.5-turbo",
    Model.claude_3: "claude-3-5-sonnet-20241022",
    Model.o1:"openai/o1-preview-2024-09-12",
}

FASTCHAT_TEMPLATE_NAMES: dict[Model, str] = {
    Model.gpt_3_5: "gpt-3.5-turbo",
    Model.gpt_4: "gpt-4o",
    Model.claude_1: "claude-instant-1.2",
    Model.claude_2: "claude-2.1",
    Model.gemini: "gemini-pro",
    Model.vicuna: "vicuna_v1.1",
    Model.llama_2: "llama-2",
    Model.llama_3: "llama-3",
    Model.mixtral: "llama-2",
    Model.qwen: "llama-2",
    Model.mistral: "mistral",
    Model.claude_3: "llama-2",
    Model.Deepseek: "llama-2",
    Model.o1: "llama-2",
}

API_KEY_NAMES: dict[Model, str] = {
    Model.gpt_3_5:  "OPENAI_API_KEY",
    Model.gpt_4:    "OPENAI_API_KEY",
    Model.claude_1: "ANTHROPIC_API_KEY",
    Model.claude_2: "ANTHROPIC_API_KEY",
    Model.gemini:   "GEMINI_API_KEY",
    Model.vicuna:   "TOGETHER_API_KEY",
    Model.llama_2:  "TOGETHER_API_KEY",
    Model.llama_3:  "TOGETHER_API_KEY",
    Model.mixtral:  "TOGETHER_API_KEY",
    Model.mistral:  "TOGETHER_API_KEY",
    Model.claude_3: "ANTHROPIC_API_KEY",
    Model.Deepseek: "TOGETHER_API_KEY",
    Model.qwen:    "TOGETHER_API_KEY",
    Model.o1: "OPENAI_API_KEY",
}


LITELLM_TEMPLATES: dict[Model, dict] = {
    Model.vicuna: {"roles":{
                    "system": {"pre_message": "", "post_message": " "},
                    "user": {"pre_message": "USER: ", "post_message": " ASSISTANT:"},
                    "assistant": {
                        "pre_message": "",
                        "post_message": "",
                    },
                },
                "post_message":"</s>",
                "initial_prompt_value" : "",
                "eos_tokens": ["</s>"]         
                },
    Model.llama_2: {"roles":{
                    "system": {"pre_message": "[INST] <<SYS>>\n", "post_message": "\n<</SYS>>\n\n"},
                    "user": {"pre_message": "", "post_message": " [/INST]"},
                    "assistant": {"pre_message": "", "post_message": ""},
                },
                "post_message" : " </s><s>",
                "initial_prompt_value" : "",
                "eos_tokens" :  ["</s>", "[/INST]"]  
            },
    
    # Model.mixtral: {"roles":{
    #                 "system": {
    #                     "pre_message": "[INST] ",
    #                     "post_message": " [/INST]"
    #                 },
    #                 "user": { 
    #                     "pre_message": "[INST] ",
    #                     "post_message": " [/INST]"
    #                 }, 
    #                 "assistant": {
    #                     "pre_message": " ",
    #                     "post_message": "",
    #                 }
    #             },
    #             "post_message": "</s>",
    #             "initial_prompt_value" : "<s>",
    #             "eos_tokens": ["</s>", "[/INST]"]
    # }
}