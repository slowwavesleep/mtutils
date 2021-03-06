import json
import random
import math
import uuid
import warnings
from dataclasses import dataclass
from typing import List, Optional

from rapidfuzz import fuzz
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from urlextract import URLExtract


@dataclass
class TranslationPair:
    idx: str
    source_name: str
    source: str
    target: str
    target_name: str
    similarity_score: Optional[float] = None

    def as_dict(self):
        return {
            "idx": self.idx,
            self.source_name: self.source,
            self.target_name: self.target
        }

    @property
    def is_source_target_identical(self):
        return self.source == self.target

    @property
    def ratio(self):
        return fuzz.ratio(self.source, self.target)

    @property
    def n_source_tokens(self):
        return len(self.source.split())

    @property
    def n_target_tokens(self):
        return len(self.target.split())

    @property
    def n_source_chars(self):
        return len(self.source)

    @property
    def n_target_chars(self):
        return len(self.target)

    @property
    def n_token_diff(self):
        return abs(self.n_target_tokens - self.n_source_tokens)

    @property
    def longest_common_prefix_len(self):
        return len(longest_common_prefix([self.source, self.target]))

    @property
    def common_prefix_ratio(self):
        return self.longest_common_prefix_len / self.n_target_chars


class TranslationDataset:

    def __init__(self, source_reader, target_reader, source_name, target_name):
        self.source_reader = source_reader
        self.target_reader = target_reader
        self.source_name = source_name
        self.target_name = target_name
        self.pairs: List[TranslationPair] = []
        self.embedder = CosineSimilarityEvaluator()
        self.similarity_evaluated = False

    def __len__(self):
        return len(self.pairs)

    def add_pair(self, pair):
        self.pairs.append(pair)

    def reset_pairs(self):
        self.pairs = []

    def read_examples(self):
        for source, target in tqdm(
                zip(
                    self.source_reader.read_examples(), self.target_reader.read_examples()
                ),
                total=self.source_reader.max_lines
        ):
            pair = TranslationPair(
                idx=str(uuid.uuid4()),
                source_name=self.source_name,
                source=source, target=target,
                target_name=self.target_name
            )
            self.add_pair(pair)

    def filter_identical(self):
        self.pairs = [pair for pair in self.pairs if not pair.is_source_target_identical]

    def filter_by_ratio(self):
        self.pairs = [pair for pair in self.pairs if pair.ratio < 80]

    def filter_by_char_len(self):
        self.pairs = [pair for pair in self.pairs if (pair.n_target_chars > 15 and pair.n_source_chars > 15)]

    def filter_by_tokens(self):
        self.pairs = [pair for pair in self.pairs if (30 > pair.n_target_tokens > 5 and 30 > pair.n_source_tokens > 5)]

    def filter_by_n_token_diff(self):
        self.pairs = [pair for pair in self.pairs if pair.n_token_diff < 3]

    def filter_by_common_ratio(self):
        self.pairs = [pair for pair in self.pairs if pair.common_prefix_ratio < 0.3]

    def filter_urls(self):
        extractor = URLExtract()
        tmp_pairs = []
        for pair in tqdm(self.pairs):
            if not (bool(extractor.find_urls(pair.source)) or bool(extractor.find_urls(pair.target))):
                tmp_pairs.append(pair)
        self.pairs = tmp_pairs

    def evaluate_pairwise_similarity(self):
        sources = [pair.source for pair in self.pairs]
        targets = [pair.target for pair in self.pairs]
        scores = self.embedder(sources, targets)

        for pair, score in zip(self.pairs, scores):
            pair.similarity_score = score

        self.similarity_evaluated = True

    def downsample(self,
                   ratio: Optional[float] = None,
                   keep_n: Optional[int] = None,
                   randomize: bool = True,
                   seed: Optional[int] = None):
        if not self.pairs:
            raise ValueError("No pairs to downsample!")
        num_to_keep = None
        if ratio is not None and keep_n is not None:
            warnings.warn("Mutually exclusive `ratio` and `keep_n` were specified. Ignoring `keep_n`...",
                          UserWarning)
        elif ratio is not None and keep_n is None:
            if 0 > ratio > 1:
                raise ValueError("Ratio must be within range 0 to 1")
            num_to_keep = math.floor(len(self) * ratio)
        elif keep_n is not None:
            if keep_n > len(self):
                warnings.warn(
                    f"`keep_n` ({keep_n}) is bigger than the current number of examples ({len(self)})."
                    f" This will not result in the reduction of the dataset size.",
                    UserWarning
                )
            num_to_keep = keep_n
        else:
            raise ValueError("Either `ratio` or `keep_n` must be specified!")
        if randomize:
            rng = random.Random(seed)
            indices_to_keep = rng.choices(range(len(self)), k=num_to_keep)
        else:
            indices_to_keep = range(num_to_keep)
        pairs = [self.pairs[i] for i in indices_to_keep]
        self.pairs = pairs

    def similarity_cutoff(self, threshold: float):
        if not self.similarity_evaluated:
            self.evaluate_pairwise_similarity()
        self.pairs = [pair for pair in self.pairs if pair.similarity_score > threshold]

    def write_to_file(self, path):
        with open(path, "w") as file:
            for pair in self.pairs:
                file.write(json.dumps(pair.as_dict(), ensure_ascii=False) + "\n")


class CosineSimilarityEvaluator:

    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/LaBSE")
        self.batch_size = 128

    def __call__(self, sentences1, sentences2):
        assert len(sentences1) == len(sentences2)

        embeddings1 = self.model.encode(sentences1, convert_to_tensor=True, show_progress_bar=True)
        embeddings2 = self.model.encode(sentences2, convert_to_tensor=True, show_progress_bar=True)

        if len(sentences1) < self.batch_size:
            cosine_scores = list(util.pytorch_cos_sim(embeddings1, embeddings2).diagonal(0).detach())
        else:
            cosine_scores = []

            for i in tqdm(range(0, len(sentences1), self.batch_size), total=len(sentences1)):
                cosine_scores.extend(
                    list(
                        util.pytorch_cos_sim(
                            embeddings1[i: i + self.batch_size], embeddings2[i: i + self.batch_size]
                        ).diagonal(0).detach()
                    )
                )

        return cosine_scores


def longest_common_prefix(strings: List[str]) -> str:
    if not strings:
        return ""
    if len(strings) == 1:
        return strings[0]

    prefix = []
    shortest = min(strings)
    longest = max(strings)
    for i in range(len(shortest)):
        short_char, long_char = shortest[i], longest[i]
        if short_char == long_char:
            prefix.append(short_char)
        else:
            break
    return "".join(prefix)
