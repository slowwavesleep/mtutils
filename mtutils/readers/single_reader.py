import json
from typing import Optional
from abc import ABC, abstractmethod


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

