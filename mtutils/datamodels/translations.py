from dataclasses import dataclass


@dataclass
class TranslationPair:
    idx: int
    source_name: str
    source: str
    target: str
    target_name: str
