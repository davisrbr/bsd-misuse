PATH_TO_DATASET = "/home/smahdi/Desktop/bsd-misuse/JB_attacks/wmdpr/datasets/wmdp_bio_decomp_filtered_41-gen.jsonl"  # Path to the running dataset file

LLAMA3_PATH = "PATH_TO_LLAMA3_MODEL"  # Path to the Llama-3 model-- used for the adaptive attack otherwise can leave it empty
LLAMA2_PATH = "PATH_TO_LLAMA2_MODEL"  # Path to the Llama-2 model-- used for adversarial reasoning otherwise can leave it empty
LLAMA3RR_PATH = "PATH_TO_LLAMA3R_MODEL"  # Path to the Llama-3-RR modell-- used for adversarial reasoning otherwise can leave it empty
R2D2_PATH = "PATH_TO_R2D2_MODEL"  # Path to the R2D2 modell-- used for adversarial reasoning otherwise can leave it empty

# PARAMETERS for PAIR ATTACK
PAIR_STREAMS = 6
PAIR_ITERS = 2

# PARAMETERS for ADAPTIVE ATTACK
ADAPT_JUDGE = 12,
PATH_TO_STRINGS_FILE = "PATH_TO_STRINGS_FILE", # Path to the adaptive attack strings file-generrated by adaptive/string_generator.ipynb

# PARAMETERS for Adversarial Reasoning Attack
ADVR_ITERS = 3
ADVR_BRANCHES = 8
ADVR_MEMORY = 32
ADVR_ATTEMPTS = 4
ADVR_K = 2

# PARAMETERS for Crescendo ATTACK
CREC_ROUNDS = 6
CREC_TRIES = 6