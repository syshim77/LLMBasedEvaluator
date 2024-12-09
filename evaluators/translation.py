import pandas as pd
from typing import List, Dict

from evaluators.base import LLMBasedEvaluator

class TranslationQualityEvaluator(LLMBasedEvaluator):
    def evaluate(self, data: List[Dict[str, str]]):
        if not data:
            raise ValueError("Input data for evaluation is empty.")

        try:
            data_df = pd.DataFrame(data)

            for row in data_df.itertuples():
                input_text = row.input_text
                translated_text = row.translated_text

        except Exception as e:
            raise Exception(f"An error occurred during evaluation: {e}")
