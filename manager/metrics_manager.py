from typing import List, Dict
from utils.metrics import accuracy, precision, recall, f1, bleu_score, rouge_score
from utils.helpers import calculate_avg

class MetricsManager:
    """
    A class for managing and computing evaluation metrics.

    Attributes:
        enable_bleu_rouge (bool): A flag to indicate whether BLEU and ROUGE score calculations are enabled. 
                                  This is relevant for translation evaluation tasks.
    """
    def __init__(self, enable_bleu_rouge: bool = False):
        self.enable_bleu_rouge = enable_bleu_rouge

    def compute_core_metrics(self, y_true: List[object], y_pred: List[object], pos_label=1, average='binary') -> Dict[str, float]:
        """
        Calculates core evaluation metrics: accuracy, precision, recall, and F1-score.

        Parameters:
            y_true (List[object]): True labels.
            y_pred (List[object]): Predicted labels.
            pos_label (Union[int, float, bool, str], optional): The label considered as positive. Defaults to 1.
            average (str, optional): The type of averaging to perform. Defaults to 'binary'.

        Returns:
            Dict[str, float]: Rounded values for accuracy, precision, recall, and F1-score.

        Raises:
            ValueError: If `y_true` or `y_pred` are empty or not the same length.
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("y_true and y_pred cannot be empty.")
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")
        
        return {
            "accuracy": round(accuracy(y_true, y_pred), 2),
            "precision": round(precision(y_true, y_pred, average=average, pos_label=pos_label), 2),
            "recall": round(recall(y_true, y_pred, average=average, pos_label=pos_label), 2),
            "f1-score": round(f1(y_true, y_pred, average=average, pos_label=pos_label), 2),
        }

    def compute_extra_metrics(self, references: List[str], translations: List[str]) -> (Dict[str, List[float]], Dict[str, float]):
        """
        Computes the BLEU and ROUGE scores for given translations.

        Parameters:
            references (List[str]): A list of reference texts. (ground truth)
            translations (List[str]): A list of translated texts.

        Returns:
            Tuple(Dict[str, List[float]], Dict[str, float]): Individual and average BLEU and ROUGE scores (scaled from 0 to 1).
          
        Raises:
            ValueError: If the input texts are empty or not the same length.
        """
        if len(references) == 0 or len(translations) == 0:
            raise ValueError("Reference and translated texts must not be empty.")
        if len(references) != len(translations):
            raise ValueError("Reference and translated texts must have the same length.")
        
        bleu_scores, rouge_1_scores, rouge_2_scores, rouge_l_scores = [], [], [], []
        for ref, trans in zip(references, translations):
            # Calculate BLEU score
            bleu_scores.append(bleu_score(ref, trans))
            
            # Calculate ROUGE scores
            rouge = rouge_score(ref, trans)
            rouge_1_scores.append(rouge["ROUGE-1"])
            rouge_2_scores.append(rouge["ROUGE-2"])
            rouge_l_scores.append(rouge["ROUGE-L"])

        return {
            "bleu": bleu_scores,
            "rouge_1": rouge_1_scores,
            "rouge_2": rouge_2_scores,
            "rouge_l": rouge_l_scores,
        }, {
            "BLEU": calculate_avg(bleu_scores),
            "ROUGE-1": calculate_avg(rouge_1_scores),
            "ROUGE-2": calculate_avg(rouge_2_scores),
            "ROUGE-L": calculate_avg(rouge_l_scores),
        }