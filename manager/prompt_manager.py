from typing import List, Dict
from settings.config import PROMPT_PATHS, TASK_REQUIREMENTS

class PromptManager:
    def __init__(self) -> None:
        """
        Initializes the PromptManager with prompt paths and task requirements.
        """
        self.prompt_paths = PROMPT_PATHS
        self.task_requirements = TASK_REQUIREMENTS

    def load_prompt(self, task: str) -> str:
        if task not in self.prompt_paths:
            raise ValueError(f"Unknown task: '{task}'. Valid tasks are: {list(self.prompt_paths.keys())}.")
        
        prompt_file = self.prompt_paths[task]
        
        try:
            with open(prompt_file, 'r') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"The prompt file for task '{task}' does not exist at: {prompt_file}.")
        except IOError as e:
            raise IOError(f"An error occurred while reading the prompt file: {e}")