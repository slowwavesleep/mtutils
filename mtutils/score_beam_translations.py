import json

from sacrebleu import sentence_ter, sentence_bleu, sentence_chrf
from bert_score import BERTScorer
from tqdm import tqdm

from models.score import ScoredExample
from readers.single_reader import LinesReader

fb_reader = LinesReader("fb_para_big.txt")
opus_reader = LinesReader("opus_para_big.txt")
para_reader = LinesReader("paracrawl_filtered_big.jsonl", json_lines=True)

bert_scorer = BERTScorer(model_type="DeepPavlov/rubert-base-cased", num_layers=12)

scored_examples = []

for fb_line, opus_line, para_line in tqdm(
        zip(
            fb_reader.read_examples(), opus_reader.read_examples(), para_reader.read_examples()
        ),
        total=57473
):
    idx = para_line["idx"]
    source = para_line["en"]
    target = para_line["ru"]
    for candidates in (fb_line, opus_line):
        for candidate in candidates.split("\t"):
            # precision, recall, f1 = bert_scorer.score([candidate], [target])
            # f1 = f1.item()
            bleu = sentence_bleu(candidate, [target]).score
            chrf = sentence_chrf(candidate, [target]).score
            ter = sentence_ter(candidate, [target]).score
            scored_example = ScoredExample(
                idx=idx,
                source=source,
                hypothesis=candidate,
                bleu=bleu,
                chrf=chrf,
                ter=ter,
                bert_score_f1=0.
            )
            scored_examples.append(scored_example)

with open("para_score_big.jsonl", "w") as file:
    for example in scored_examples:
        file.write(json.dumps(example.as_dict(), ensure_ascii=False) + "\n")
