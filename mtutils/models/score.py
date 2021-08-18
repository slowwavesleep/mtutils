from dataclasses import dataclass


@dataclass
class BaseScore:
    idx: str
    score: float


@dataclass
class CosineSimilarityScore:
    pass
