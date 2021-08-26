import json
from typing import Optional, Union, Sequence
from abc import ABC, abstractmethod

from lxml.etree import iterparse


class BaseReader(ABC):

    @abstractmethod
    def read_examples(self):
        ...


class LinesReader(BaseReader):
    def __init__(
            self,
            path: str,
            max_lines: Optional[int] = None,
            skip_first_n: Optional[int] = None,
            json_lines: bool = False
    ):
        self.path = path
        self.max_lines = max_lines
        self.skip_first_n = skip_first_n
        self.json_lines = json_lines

    def read_examples(self):
        with open(self.path) as file:
            for i, line in enumerate(file):
                if self.max_lines and i >= self.max_lines:
                    return
                if not (self.skip_first_n and i in range(self.skip_first_n)):
                    if self.json_lines:
                        yield json.loads(line.strip("\n"))
                    else:
                        yield line.strip("\n")


class XmlLinesReader(BaseReader):

    def __init__(self, path: str, tags_to_keep: Union[str, Sequence[str]]):
        self.path = path
        if isinstance(tags_to_keep, str):
            self.tags_to_keep = [tags_to_keep]
        else:
            self.tags_to_keep = tags_to_keep

    def read_examples(self):
        iter_tree = iterparse(self.path)
        for _, element in iter_tree:
            tag = element.tag
            if tag in self.tags_to_keep:
                yield element.text
