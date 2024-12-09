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
                pass

            return {
                "individual_results": individual_results,
                "overall_results": overall_results
            }

        except Exception as e:
            raise Exception(f"An error occurred during evaluation: {e}")


    def get_quality_confidence(self, pattern: str, source_text: str) -> (str, float):
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