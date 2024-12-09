from typing import List, Dict
from settings.config import PROMPT_PATHS, TASK_REQUIREMENTS

class PromptManager:
    """
    A class to manage and format prompts for various tasks.

    Attributes:
        prompt_paths (dict): A dictionary mapping task names to file paths containing prompts.
        task_requirements (dict): A dictionary defining the number of required contents for each task.
    """

    def __init__(self) -> None:
        """
        Initializes the PromptManager with prompt paths and task requirements.
        """
        self.prompt_paths = PROMPT_PATHS
        self.task_requirements = TASK_REQUIREMENTS

    def load_prompt(self, task: str) -> str:
        """
        Loads the system prompt for a specified task.

        Parameters:
            task (str): The task type ('review', 'translation', or 'translation_eval').

        Returns:
            str: The content of the system prompt file.

        Raises:
            ValueError: If the task is not recognized.
            FileNotFoundError: If the prompt file does not exist.
            IOError: If there is an issue reading the file.
        """
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

    def format_prompt(self, task: str, contents: List[str]) -> List[Dict[str, str]]:
        """
        Formats prompts based on the task and provided content.

        Parameters:
            task (str): The task type ('review', 'translation', or 'translation_eval').
            contents (List[str]): List of input strings for the task.

        Returns:
            List[Dict[str, str]]: A formatted prompt as a list of dictionaries.

        Raises:
            ValueError: 
                - If the task is not recognized.
                - If no content is provided.
                - If the number of contents exceeds the allowed maximum for the task.
        """
        if not contents:
            raise ValueError("Please provide at least one content.")
        
        if task not in self.task_requirements:
            raise ValueError(f"Unknown task: '{task}'. Valid tasks are: {list(self.task_requirements.keys())}.")

        max_contents = self.task_requirements[task]

        if len(contents) > max_contents:
            raise ValueError(
                f"Too many contents for the '{task}' task. Max allowed: {max_contents}, provided: {len(contents)}."
            )
        
        system_prompt = self.load_prompt(task)

        # Prepare the base prompt
        formatted_prompt = [{"role": "system", "content": system_prompt}]
        
        # Format user-specific content
        if task in ['review', 'translation']:  # Similar formatting for these tasks
            formatted_prompt.append({"role": "user", "content": contents[0]})
        elif task == 'translation_eval':  # Special format for translation evaluation
            formatted_prompt.append({
                "role": "user",
                "content": f"input_text: {contents[0]}, translated_text: {contents[1]}"
            })

        return formatted_prompt