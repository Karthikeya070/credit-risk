#src/scoring.py
import numpy as np

PDO = 20
BASE_SCORE = 600
BASE_ODDS = 50

FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR*np.log(BASE_ODDS)

def probability_to_score(prob):
    odds = (1 - prob) / prob
    score = OFFSET + FACTOR * np.log(odds)
    return int(score)