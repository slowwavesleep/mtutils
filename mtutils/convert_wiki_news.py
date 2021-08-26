from readers.single_reader import XmlLinesReader

source_reader = XmlLinesReader(path="data/newstest2020-enru-src.en.sgm", tag_to_keep="<seg")
target_reader = XmlLinesReader(path="data/newstest2020-enru-ref.ru.sgm", tag_to_keep="<seg")

with open("newstest2020_test_en.txt", "w") as file:
    for line in source_reader.read_examples():
        file.write(line + "\n")


with open("newstest2020_test_ru.txt", "w") as file:
    for line in target_reader.read_examples():
        file.write(line + "\n")
