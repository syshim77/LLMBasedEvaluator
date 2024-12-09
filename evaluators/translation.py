import pandas as pd
from typing import List, Dict

from utils.helpers import find_pattern, calculate_avg
from evaluators.base import LLMBasedEvaluator

class TranslationQualityEvaluator(LLMBasedEvaluator):
    def evaluate(self, data: List[Dict[str, str]]):
        if not data:
            raise ValueError("Input data for evaluation is empty.")

        try:
            y_true, confidence_scores, individual_results, overall_results = [], [], [], []
            data_df = pd.DataFrame(data)
            
            if not {'input_text', 'translated_text', 'quality'}.issubset(data_df.columns):
                raise ValueError("Input data must contain 'input_text', 'translated_text', and 'quality' fields.")

            for row in data_df.itertuples():
                # Format the prompt for translation evaluation
                prompt = self.prompt_manager.format_prompt('translation_eval', [row.input_text, row.translated_text])

                # Perform inference
                result = self.inference(prompt)
                
            # Compute additional metrics if required
            if self.metrics_manager.enable_bleu_rouge:
                pass

            return {
                "individual_results": individual_results,
                "overall_results": overall_results
            }

        except Exception as e:
            raise Exception(f"An error occurred during evaluation: {e}")
