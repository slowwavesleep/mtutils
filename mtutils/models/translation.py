from dataclasses import dataclass
import json


@dataclass
class TranslationPair:
    idx: int
    source_name: str
    source: str
    target: str
    target_name: str

    def as_dict(self):
        return vars(self)
        return json.dumps(self, ensure_ascii=False)

