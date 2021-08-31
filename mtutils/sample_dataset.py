from readers.single_reader import LinesReader
from models.translation import TranslationDataset

source_path = "data/paracrawl-release1.en-ru.zipporah0-dedup-clean.en"
target_path = "data/paracrawl-release1.en-ru.zipporah0-dedup-clean.ru"

source_reader = LinesReader(source_path, max_lines=1_000_000)
target_reader = LinesReader(target_path, max_lines=1_000_000)

dataset = TranslationDataset(
    source_reader=source_reader, target_reader=target_reader, source_name="en", target_name="ru"
)

dataset.read_examples()
dataset.filter_identical()
dataset.filter_by_ratio()
dataset.filter_by_tokens()
dataset.filter_by_char_len()
dataset.filter_by_n_token_diff()
dataset.filter_by_common_ratio()
dataset.downsample(keep_n=100_000, seed=42)
dataset.filter_urls()
dataset.similarity_cutoff(0.93)
dataset.write_to_file("paracrawl_filtered_big.jsonl")
