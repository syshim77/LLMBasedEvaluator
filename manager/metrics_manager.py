from typing import List, Dict
from utils.metrics import accuracy, precision, recall, f1, bleu_score, rouge_score

class MetricsManager:
    def __init__(self, enable_bleu_rouge: bool = False):
        self.enable_bleu_rouge = enable_bleu_rouge

    def compute_core_metrics(self, y_true: List[object], y_pred: List[object], pos_label=1, average='binary') -> Dict[str, float]:
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