import re
from typing import List

def find_pattern(pattern: str, source_text: str) -> List[object]:
    return re.search(pattern, source_text, re.IGNORECASE)


def calculate_avg(x: List[float]) -> float:
    if not x:
        raise ValueError("The input list must not be empty.")
    return round(sum(x) / len(x), 2)
