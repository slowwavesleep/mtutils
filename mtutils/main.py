from readers.single_reader import TxtReader
from models.translation import TranslationPair

source_path = "data/reddit_dev.en"
target_path = "data/reddit_dev.ru"

source_reader = TxtReader(source_path)
target_reader = TxtReader(target_path)

for i, (source, target) in enumerate(zip(source_reader.read_examples(), target_reader.read_examples())):
    pair = TranslationPair(idx=i, source_name="ru", source=source, target=target, target_name="en")
    print(pair.as_dict())
