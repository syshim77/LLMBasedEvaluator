import pandas as pd
from typing import List, Dict
from evaluators.base import LLMBasedEvaluator

class SentimentReviewEvaluator(LLMBasedEvaluator):
    def evaluate(self, data: List[Dict[str, str]]):
        if not data:
            raise ValueError("Input data for evaluation is empty.")

        try:
            y_true, confidence_scores, individual_results, overall_results = [], [], [], []
            data_df = pd.DataFrame(data)

            if 'text' not in data_df.columns or 'label' not in data_df.columns:
                raise ValueError("Input data must contain 'text' and 'label' fields.")

            for row in data_df.itertuples():
                # Format the prompt for the LLM
                prompt = self.prompt_manager.format_prompt('review', [row.text])

                # Perform inference
                result = self.inference(prompt)

            return {
                "individual_results": individual_results,
                "overall_results": overall_results
            }
        
        except Exception as e:
            raise Exception(f"An error occurred during evaluation: {e}")
