import torch
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMBasedEvaluator:
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.tokenizer, self.model = self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is expected but not available on this machine.")
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name).to(device)
            return tokenizer, model
        except Exception as e:
            raise Exception(f"Failed to load model or tokenizer for '{self.model_name}': {e}")

    def __call__(self, data: List[Dict[str, object]]) -> None:
        return self.evaluate(data)

    def evaluate(self, data: List[Dict[str, object]]) -> None:
        raise NotImplementedError("Subclasses must implement this method.")