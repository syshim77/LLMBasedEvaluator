import torch
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

from manager.prompt_manager import PromptManager
from manager.metrics_manager import MetricsManager

class LLMBasedEvaluator:
    """
    A class to evaluate tasks using a language model (LLM) based approach.

    Attributes:
        model_name (str): The name of the pretrained language model.
        tokenizer (AutoTokenizer): Tokenizer for the specified model.
        model (AutoModelForCausalLM): Pretrained language model.
        prompt_manager (PromptManager): Manages prompts for different tasks.
        metrics_manager (MetricsManager): MetricsManager instance for managing metrics.
    """

    def __init__(self, model_name: str, metrics_manager: MetricsManager) -> None:
        """
        Initializes the evaluator with the specified model and metrics type.

        Parameters:
            model_name (str): The name of the pretrained language model.
            metrics_manager (MetricsManager): MetricsManager instance.

        Raises:
            ValueError: If an unsupported metrics type is provided.
        """
        super().__init__()
        self.model_name = model_name
        self.tokenizer, self.model = self.load_model_and_tokenizer()
        self.prompt_manager = PromptManager()
        self.metrics_manager = metrics_manager


    def __call__(self, data: List[Dict[str, object]]) -> None:
        """
        Enables the class to be called like a function, delegating to the `evaluate` method.

        Parameters:
            data (List[Dict[str, object]]): The data to be evaluated.
        """
        return self.evaluate(data)


    def load_model_and_tokenizer(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        Loads the tokenizer and model for the specified pretrained LLM.

        Returns:
            Tuple[AutoTokenizer, AutoModelForCausalLM]: The tokenizer and model objects.

        Raises:
            RuntimeError: If CUDA is not available when expected.
            Exception: For issues during model or tokenizer loading.
        """
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is expected but not available on this machine.")
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name).to(device)
            return tokenizer, model
        except Exception as e:
            raise Exception(f"Failed to load model or tokenizer for '{self.model_name}': {e}")


    def inference(self, prompt: List[Dict[str, str]]) -> str:
        """
        Performs inference using the model on a given prompt.

        Parameters:
            prompt (List[Dict[str, str]]): The prompt formatted as a list of dictionaries.

        Returns:
            str: The generated output text.

        Raises:
            ValueError: If the prompt is not in the expected format.
            Exception: For errors during tokenization or generation.
        """
        if not isinstance(prompt, list) or not all(isinstance(p, dict) for p in prompt):
            raise ValueError("Prompt must be a list of dictionaries.")
        
        try:
            tokenized_prompt = self.tokenizer.apply_chat_template(
                prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)
            
            generated_output = self.model.generate(tokenized_prompt, max_new_tokens=50)
            output = self.tokenizer.decode(generated_output[0], skip_special_tokens=True)

            return output.split("assistant\n\n")[-1]
        except Exception as e:
            raise Exception(f"Error during inference: {e}")
        

    def evaluate(self, data: List[Dict[str, object]]) -> None:
        """
        Placeholder for the evaluation method.
        """
        raise NotImplementedError("Subclasses must implement this method.")