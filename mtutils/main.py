from readers.single_reader import TxtReader
from models.translation import TranslationDataset

source_path = "data/paracrawl-release1.en-ru.zipporah0-dedup-clean.en"
target_path = "data/paracrawl-release1.en-ru.zipporah0-dedup-clean.ru"

source_reader = TxtReader(source_path, max_lines=500_000)
target_reader = TxtReader(target_path, max_lines=500_000)

dataset = TranslationDataset(
    source_reader=source_reader, target_reader=target_reader, source_name="en", target_name="ru"
)

dataset.read_examples()
dataset.filter_identical()
dataset.filter_urls()
dataset.filter_by_ratio()
dataset.filter_by_tokens()
dataset.filter_by_char_len()
dataset.filter_by_n_token_diff()
dataset.filter_by_common_ratio()
dataset.downsample(keep_n=7000, seed=42)
dataset.similarity_cutoff(0.87)
dataset.write_to_file("paracrawl_filtered.jsonl")
