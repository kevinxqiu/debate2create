from pathlib import Path


def file_to_string(filename):
    return Path(filename).read_text()
