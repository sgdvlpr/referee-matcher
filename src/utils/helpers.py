import re

def abstract_index_to_text(abstract_index: dict) -> str:
    """
    Convert a reversed abstract index (word -> list of positions)
    back to a plain abstract text string, ordered by positions,
    cleaned of special characters like \r and \n.

    Parameters:
        abstract_index (dict): keys are words, values are lists of int positions.

    Returns:
        str: reconstructed plain abstract text.
    """
    # Create a map position -> word for all positions
    pos_word_map = {}
    for word, positions in abstract_index.items():
        for pos in positions:
            pos_word_map[pos] = word

    # Sort positions to get words in order
    sorted_positions = sorted(pos_word_map.keys())

    # Join words by their sorted position
    words_in_order = [pos_word_map[pos] for pos in sorted_positions]
    reconstructed = ' '.join(words_in_order)

    # Clean special characters \r, \n, multiple spaces
    cleaned = re.sub(r'[\r\n]+', ' ', reconstructed)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned

abstract_index = {
    "This": [0],
    "paper": [1],
    "presents": [2],
    "a": [3, 16, 22, 27, 70, 96, 122, 127],
    "study": [4],
    "of": [5, 15, 30, 35, 83, 107, 121, 138],
    "phase": [6, 44, 48, 118],
    "noise": [7, 23, 49, 84, 119],
    "in": [8, 85, 95],
    "two": [9, 92],
    "inductorless": [10],
    "CMOS": [11, 36, 71, 99],
    "oscillators.": [12],
    "First-order": [13],
    "analysis": [14],
    "linear": [17, 33],
    "oscillatory": [18],
    "system": [19],
    "leads": [20],
    "to": [21, 41, 81, 103],
    "shaping": [24],
    "function": [25],
    "and": [26, 46, 57, 63, 78, 91, 126],
    "new": [28],
    "definition": [29],
    "Q.": [31],
    "A": [32],
    "model": [34],
    "ring": [37, 124],
    "oscillators": [38],
    "is": [39, 74],
    "used": [40, 102],
    "calculate": [42],
    "their": [43],
    "noise,": [45, 53, 56, 60],
    "three": [47],
    "phenomena,": [50],
    "namely,": [51],
    "additive": [52],
    "high-frequency": [54],
    "multiplicative": [55, 59],
    "low-frequency": [58],
    "are": [61, 89, 101],
    "identified": [62],
    "formulated.": [64],
    "Based": [65],
    "on": [66],
    "the": [67, 86, 105, 108, 113, 116],
    "same": [68],
    "concepts,": [69],
    "relaxation": [72, 129],
    "oscillator": [73, 125, 130],
    "also": [75],
    "analyzed.": [76],
    "Issues": [77],
    "techniques": [79],
    "related": [80],
    "simulation": [82],
    "time": [87],
    "domain": [88],
    "described,": [90],
    "prototypes": [93],
    "fabricated": [94],
    "0.5-/spl": [97],
    "mu/m": [98],
    "technology": [100],
    "investigate": [104],
    "accuracy": [106],
    "theoretical": [109],
    "predictions.": [110],
    "Compared": [111],
    "with": [112],
    "measured": [114],
    "results,": [115],
    "calculated": [117],
    "values": [120],
    "2-GHz": [123],
    "900-MHz": [128],
    "at": [131],
    "5": [132],
    "MHz": [133],
    "offset": [134],
    "have": [135],
    "an": [136],
    "error": [137],
    "approximately": [139],
    "4": [140],
    "dB.": [141]
  }

plain_text = abstract_index_to_text(abstract_index)
print(plain_text)