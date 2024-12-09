import pandas as pd
from typing import List, Dict
from utils.helpers import find_pattern, calculate_avg
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
    
    
    def get_label_confidence(self, pattern: str, source_text: str) -> (str, float):
        try:
            match = find_pattern(pattern, source_text)
            if match:
                if len(match.groups()) == 2:
                    label = match.group(1).lower()
                    confidence = float(match.group(2))
                elif float(match.group(1)):  # only confidence score is generated
                    label = None
                    confidence = float(match.group(1))
                elif match.group(1).lower in ('positive', 'negative'): # only label is generated
                    label = match.group(1).lower()
                    confidence = 0.0
                else:
                    label, confidence = None, 0.0
            else:
                label, confidence = None, 0.0
                print("No valid pattern found in the source text.")

            return label, confidence

        except ValueError as ve:
            raise ve
        except Exception as e:
            raise Exception(f"An unexpected error occurred during label and confidence extraction: {e}")
