from readers.single_reader import TxtReader
from models.translation import TranslationPair, TranslationDataset

source_path = "data/paracrawl-release1.en-ru.zipporah0-dedup-clean.en"
target_path = "data/paracrawl-release1.en-ru.zipporah0-dedup-clean.ru"

source_reader = TxtReader(source_path, max_lines=20_000)
target_reader = TxtReader(target_path, max_lines=20_000)

dataset = TranslationDataset([])

for i, (source, target) in enumerate(zip(source_reader.read_examples(), target_reader.read_examples())):
    pair = TranslationPair(idx=i, source_name="ru", source=source, target=target, target_name="en")
    dataset.add_pair(pair)

dataset.downsample(0.25)
dataset.write_to_file("check.jsonl")
