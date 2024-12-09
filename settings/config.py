# Paths to the JSON data files.
DATA_PATHS = {
    "review": "./data/review.json",
    "translation": "./data/translation.json"
}

# Paths to the pre-defined system prompt files.
PROMPT_PATHS = {
    "review": "./data/system_prompts/review_prompt.txt",
    "translation": "./data/system_prompts/translation_prompt.txt",
    "translation_eval": "./data/system_prompts/translation_eval_prompt.txt"
}

# The number of required contents for each task.
TASK_REQUIREMENTS = {
  "review": 1,
  "translation": 1,
  "translation_eval": 2,
}

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"