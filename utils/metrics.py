from typing import List, Union, Literal
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sacrebleu.metrics import BLEU
from rouge import Rouge

def accuracy(y_true: list, y_pred: list) -> float:
    """
    Computes the accuracy of predictions.

    Parameters:
        y_true (list): The ground truth labels.
        y_pred (list): The predicted labels.

    Returns:
        float: Accuracy score.
        
    Raises:
        ValueError: If the input lists are empty or have mismatched lengths.
    """
    if not y_true or not y_pred:
        raise ValueError("Input lists y_true and y_pred must not be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError("Input lists y_true and y_pred must have the same length.")
    return accuracy_score(y_true, y_pred)


def precision(
    y_true: list, 
    y_pred: list, 
    pos_label: Union[int, float, bool, str] = 1, 
    average: Literal["micro", "macro", "samples", "weighted", "binary", None] = 'binary'
) -> float:
    """
    Computes the precision of predictions.

    Parameters:
        y_true (list): The ground truth labels.
        y_pred (list): The predicted labels.
        pos_label (Union[int, float, bool, str]): The label considered as positive.
        average (str): The averaging method for multiclass tasks.

    Returns:
        float: Precision score.
        
    Raises:
        ValueError: If the input lists are empty or have mismatched lengths.
    """
    if not y_true or not y_pred:
        raise ValueError("Input lists y_true and y_pred must not be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError("Input lists y_true and y_pred must have the same length.")
    return precision_score(y_true, y_pred, pos_label=pos_label, average=average)


def recall(
    y_true: list, 
    y_pred: list, 
    pos_label: Union[int, float, bool, str] = 1, 
    average: Literal["micro", "macro", "samples", "weighted", "binary", None] = 'binary'
) -> float:
    """
    Computes the recall of predictions.

    Parameters:
        y_true (list): The ground truth labels.
        y_pred (list): The predicted labels.
        pos_label (Union[int, float, bool, str]): The label considered as positive.
        average (str): The averaging method for multiclass tasks.

    Returns:
        float: Recall score.
        
    Raises:
        ValueError: If the input lists are empty or have mismatched lengths.
    """
    if not y_true or not y_pred:
        raise ValueError("Input lists y_true and y_pred must not be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError("Input lists y_true and y_pred must have the same length.")
    return recall_score(y_true, y_pred, pos_label=pos_label, average=average)


def f1(
    y_true: list, 
    y_pred: list, 
    pos_label: Union[int, float, bool, str] = 1, 
    average: Literal["micro", "macro", "samples", "weighted", "binary", None] = 'binary'
) -> float:
    """
    Computes the F1 score of predictions.

    Parameters:
        y_true (list): The ground truth labels.
        y_pred (list): The predicted labels.
        pos_label (Union[int, float, bool, str]): The label considered as positive.
        average (str): The averaging method for multiclass tasks.

    Returns:
        float: F1 score.
        
    Raises:
        ValueError: If the input lists are empty or have mismatched lengths.
    """
    if not y_true or not y_pred:
        raise ValueError("Input lists y_true and y_pred must not be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError("Input lists y_true and y_pred must have the same length.")
    return f1_score(y_true, y_pred, pos_label=pos_label, average=average)


def bleu_score(reference_text: str, translated_text: str) -> float:
    """
    Computes the BLEU score for a given translation.

    Parameters:
        reference_text (str): The reference text. (ground truth)
        translated_text (str): The translated text.

    Returns:
        float: BLEU score (scaled from 0 to 1).
        
    Raises:
        ValueError: If the input texts are empty.
    """
    if not reference_text or not translated_text:
        raise ValueError("Reference and translated texts must not be empty.")
    bleu_scorer = BLEU(effective_order=True)
    score = bleu_scorer.sentence_score(hypothesis=translated_text, references=[reference_text])
    return score.score / 100  # sacreBLEU gives the score in percent


def rouge_score(reference_text: str, translated_text: str) -> dict:
    """
    Computes the ROUGE scores for a given translation.

    Parameters:
        reference_text (str): The reference text. (ground truth)
        translated_text (str): The translated text.

    Returns:
        dict: A dictionary with ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
        
    Raises:
        ValueError: If the input texts are empty.
    """
    if not reference_text or not translated_text:
        raise ValueError("Reference and translated texts must not be empty.")
    rouge_scorer = Rouge()
    score = rouge_scorer.get_scores(hyps=translated_text, refs=reference_text)
    return {
        "ROUGE-1": score[0]["rouge-1"]["f"],
        "ROUGE-2": score[0]["rouge-2"]["f"],
        "ROUGE-L": score[0]["rouge-l"]["f"]
    }