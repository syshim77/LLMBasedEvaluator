import pandas as pd
from typing import List, Dict
from utils.helpers import find_pattern, calculate_avg
from evaluators.base import LLMBasedEvaluator

class SentimentReviewEvaluator(LLMBasedEvaluator):
    """
    A class to evaluate sentiment review tasks using a language model (LLM).

    This class extends `LLMBasedEvaluator` and implements the evaluation method for sentiment reviews.

    Methods:
        evaluate(data): Evaluates sentiment reviews by comparing predicted labels to true labels.
        get_label_confidence(pattern, source_text): Extracts the predicted label and confidence score from the LLM output.
    """

    def evaluate(self, data: List[Dict[str, str]]) -> Dict[str, object]:
        """
        Evaluates the sentiment review data and computes core metrics.

        Parameters:
            data (List[Dict[str, str]]): List of dictionaries containing input texts and true labels.

        Returns:
            Dict[str, object]: Evaluation results including accuracy, precision, recall, F1-score, and average confidence.

        Raises:
            ValueError: If input data is empty or improperly formatted.
            Exception: For unexpected errors during evaluation.
        """
        if not data:
            raise ValueError("Input data for evaluation is empty.")

        try:
            y_true, confidence_scores, individual_results = [], [], []
            data_df = pd.DataFrame(data)

            if 'text' not in data_df.columns or 'label' not in data_df.columns:
                raise ValueError("Input data must contain 'text' and 'label' fields.")

            for row in data_df.itertuples():
                # Format the prompt for the LLM
                prompt = self.prompt_manager.format_prompt('review', [row.text])

                # Perform inference
                result = self.inference(prompt)

                # Extract label and confidence
                pattern = r'\b(positive|negative)[,\s]*([0-1](?:\.\d+)?)'
                label, confidence = self.get_label_confidence(pattern, result)
                ind_result = {"text":row.text, "label":row.label, "evaluated_label":label, "confidence":confidence}

                y_true.append(label)
                confidence_scores.append(confidence)
                individual_results.append(ind_result)                

            # Extract true labels and calculate metrics
            y_pred = data_df['label'].values.tolist()
            avg_confidence = calculate_avg(confidence_scores)
            overall_results = self.metrics_manager.compute_core_metrics(y_true, y_pred, pos_label='positive')
            overall_results["avg_confidence"] = avg_confidence

            return {
                "individual_results": individual_results,
                "overall_results": overall_results
            }

        except Exception as e:
            raise Exception(f"An error occurred during evaluation: {e}")

    def get_label_confidence(self, pattern: str, source_text: str) -> (str, float):
        """
        Extracts the sentiment label and confidence score from the LLM-generated output.

        Parameters:
            pattern (str): A regex pattern to match the label and confidence.
            source_text (str): The LLM-generated text to parse.

        Returns:
            Tuple[str, float]: The sentiment label ('positive' or 'negative') and the confidence score.
        """
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