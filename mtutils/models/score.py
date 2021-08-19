from dataclasses import dataclass


@dataclass
class ScoredExample:
    idx: str
    source: str
    hypothesis: str
    bleu: float
    chrf: float
    ter: float
    bert_score_f1: float

    def as_dict(self):
        return vars(self)

