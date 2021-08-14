from typing import List, Optional


class BaseReader:
    pass


class TxtReader(BaseReader):
    def __init__(self, path: str, max_lines: Optional[int] = None, skip_first_n: Optional[int] = None):
        self.path = path
        self.max_lines = max_lines
        self.skip_first_n = skip_first_n

    def read_examples(self):
        with open(self.path) as file:
            for i, line in enumerate(file):
                if self.max_lines and i > self.max_lines:
                    raise StopIteration
                if not (self.skip_first_n and i in range(self.skip_first_n)):
                    yield line.strip("\n")