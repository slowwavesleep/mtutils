import json
import random
import math

from dataclasses import dataclass
from typing import List


@dataclass
class TranslationPair:
    idx: int
    source_name: str
    source: str
    target: str
    target_name: str

    def as_dict(self):
        return {
            "idx": self.idx,
            self.source_name: self.source,
            self.target_name: self.target,
        }


class TranslationDataset:

    def __init__(self, pairs: List[TranslationPair]):
        self.pairs: List[TranslationPair] = pairs

    def __len__(self):
        return len(self.pairs)

    def add_pair(self, pair):
        self.pairs.append(pair)

    def downsample(self, ratio: float):
        if 0 > ratio > 1:
            raise ValueError("Ratio must be within range 0 to 1")
        num_to_keep = math.floor(self.__len__() * ratio)
        indices_to_keep = random.choices(range(self.__len__()), k=num_to_keep)
        pairs = [self.pairs[i] for i in indices_to_keep]
        self.pairs = pairs

    def write_to_file(self, path):
        with open(path, "w") as file:
            for pair in self.pairs:
                file.write(json.dumps(pair.as_dict(), ensure_ascii=False) + "\n")

