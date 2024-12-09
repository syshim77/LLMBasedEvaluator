import pandas as pd
from typing import List, Dict

from utils.helpers import find_pattern, calculate_avg
from evaluators.base import LLMBasedEvaluator

class TranslationQualityEvaluator(LLMBasedEvaluator):
    """
    A class to evaluate translation quality using a language model (LLM).

    This class extends `LLMBasedEvaluator` and evaluates translation tasks by computing 
    core metrics and additional translation-related metrics (BLEU and ROUGE scores).

    Methods:
        evaluate(data): Evaluates translation quality and computes metrics.
        translate(data): Generates translations and computes BLEU and ROUGE scores.
        get_translated_text(pattern, source_text): Extracts the translated text from LLM-generated output.
        get_quality_confidence(pattern, source_text): Extracts quality ratings and confidence scores from LLM output.
    """

    def evaluate(self, data: List[Dict[str, str]]) -> Dict[str, object]:
        """
        Evaluates the translation quality and computes metrics.

        Parameters:
            data (List[Dict[str, str]]): A list of dictionaries containing input text, 
                                         true quality ratings, and translated text.

        Returns:
            Dict[str, object]: Metrics including accuracy, precision, recall, F1-score, average confidence, 
                               and optionally BLEU/ROUGE scores.

        Raises:
            ValueError: If input data is empty or improperly formatted.
            Exception: For unexpected errors during evaluation.
        """
        if not data:
            raise ValueError("Input data for evaluation is empty.")

        try:
            y_true, confidence_scores, individual_results = [], [], []
            data_df = pd.DataFrame(data)

            if not {'input_text', 'translated_text', 'quality'}.issubset(data_df.columns):
                raise ValueError("Input data must contain 'input_text', 'translated_text', and 'quality' fields.")

            for row in data_df.itertuples():
                # Format the prompt for translation evaluation
                prompt = self.prompt_manager.format_prompt('translation_eval', [row.input_text, row.translated_text])

                # Perform inference
                result = self.inference(prompt)

                pattern = r'(?i)\b(low|medium|high)\b.*?\b([0-9]*\.[0-9]+)\b'
                quality, confidence = self.get_quality_confidence(pattern, result)
                ind_result = {
                    "input_text": row.input_text,
                    "translated_text": row.translated_text,
                    "quality": row.quality,
                    "evaluated_quality": quality,
                    "confidence": confidence
                }

                y_true.append(quality)
                confidence_scores.append(confidence)
                individual_results.append(ind_result)

            # Compute core metrics
            y_pred = data_df["quality"].values.tolist()
            avg_confidence = calculate_avg(confidence_scores)
            overall_results = self.metrics_manager.compute_core_metrics(y_true, y_pred, average="macro")
            overall_results["avg_confidence"] = avg_confidence

            # Compute additional metrics if required
            if self.metrics_manager.enable_bleu_rouge:
                ind_extra_results, avg_extra_results = self.translate(data)
                individual_results = [
                    {**ind_res, "bleu": ind_bleu, "rouge_1": ind_rouge_1, "rouge_2": ind_rouge_2, "rouge_l": ind_rouge_l}
                    for ind_res, ind_bleu, ind_rouge_1, ind_rouge_2, ind_rouge_l in zip(individual_results, ind_extra_results["bleu"], ind_extra_results["rouge_1"], ind_extra_results["rouge_2"], ind_extra_results["rouge_l"])
                ]
                overall_results.update(avg_extra_results)

            return {
                "individual_results": individual_results,
                "overall_results": overall_results
            }

        except Exception as e:
            raise Exception(f"An error occurred during evaluation: {e}")

    def translate(self, data: List[Dict[str, str]]) -> (Dict[str, List[float]], Dict[str, float]):
        """
        Generates translations and computes BLEU and ROUGE scores.

        Parameters:
            data (List[Dict[str, str]]): A list of dictionaries containing input text and translated text.

        Returns:
            Tuple(Dict[str, List[float]], Dict[str, float]):
                Individual BLEU and ROUGE scores, Average BLEU and ROUGE scores.

        Raises:
            Exception: For unexpected errors during translation or score computation.
        """
        try:
            data_df = pd.DataFrame(data)
            references, translated_texts = [], []
            for row in data_df.itertuples():
                # Format the prompt for translation
                prompt = self.prompt_manager.format_prompt('translation', [row.input_text])

                # Perform inference
                result = self.inference(prompt)

                # Extract the translated text
                pattern = r'^[^\n]+(?=\n|$)'
                translation = self.get_translated_text(pattern, result)
                references.append(translation)
                translated_texts.append(row.translated_text)

            ind_extra_results, avg_extra_results = self.metrics_manager.compute_extra_metrics(references, translated_texts)
            return ind_extra_results, avg_extra_results

        except Exception as e:
            raise Exception(f"An error occurred during translation or score computation: {e}")

    def get_translated_text(self, pattern: str, source_text: str) -> str:
        """
        Extracts the translated text from the LLM-generated output.

        Parameters:
            pattern (str): A regex pattern to match the translation.
            source_text (str): The LLM-generated output.

        Returns:
            str: The extracted translation.
        """
        try:
            match = find_pattern(pattern, source_text)
            if match:
                return match.group(0)
            else:
                print("No valid pattern found in the source text.")
                return None
        except Exception as e:
            raise Exception(f"An error occurred while extracting translated text: {e}")

    def get_quality_confidence(self, pattern: str, source_text: str) -> (str, float):
        """
        Extracts the quality rating and confidence score from the LLM-generated output.

        Parameters:
            pattern (str): A regex pattern to match the quality and confidence.
            source_text (str): The LLM-generated output.

        Returns:
            Tuple[str, float]: The quality rating ('low', 'medium', 'high') and confidence score.

        Raises:
            ValueError: If the pattern does not match two numbers.
        """
        try:
            match = find_pattern(pattern, source_text)
            if match:
              if len(match.groups()) == 2:
                  quality = match.group(1).lower()
                  confidence = float(match.group(2))
              elif float(match.group(1)): # only confidence score is generated
                  quality = None
                  confidence = float(match.group(1))
              elif match.group(1).lower() in ('low', 'medium', 'high'): # only quality is generated
                  quality = match.group(1).lower()
                  confidence = 0.0
              else:
                  quality, confidence = None, 0.0
            else:
                quality, confidence = None, 0.0
                print("No match found")

            return quality, confidence

        except KeyError:
            raise ValueError(f"Invalid quality rating found in the source text: {match.groups()}")
        except Exception as e:
            raise Exception(f"An error occurred while extracting quality and confidence: {e}")