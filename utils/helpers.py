import re
from typing import List

def find_pattern(pattern: str, source_text: str) -> List[object]:
    """
    Finds matches of a regular expression pattern in the provided text.

    Parameters:
        pattern (str): The regex pattern to search for.
        source_text (str): The text in which to search for the pattern.

    Returns:
        List[object]: A list of matches found, considering case-insensitive matching.
    """
    return re.search(pattern, source_text, re.IGNORECASE)


def calculate_avg(x: List[float]) -> float:
    """
    Calculates the average of a list of numeric values.

    Parameters:
        x (List[float]): A list of numeric values.

    Returns:
        float: The average of the values rounded to two decimal places.

    Raises:
        ValueError: If the list is empty.
    """
    if not x:
        raise ValueError("The input list must not be empty.")
    return round(sum(x) / len(x), 2)
