import torch
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMBasedEvaluator:
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name

    def __call__(self, data: List[Dict[str, object]]) -> None:
        return self.evaluate(data)

    def evaluate(self, data: List[Dict[str, object]]) -> None:
        raise NotImplementedError("Subclasses must implement this method.")