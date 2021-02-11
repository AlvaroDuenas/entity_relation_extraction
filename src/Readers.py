import codecs
import os
import re
from typing import List

from Entities import Dataset


class Standoff:
    annotation_extensions = ['ann', 'a1', 'a2']
    separators = ['Title', 'Paragraph']
    split_paragraphs = False
    

    @staticmethod
    def load(file_path: str) -> Dataset:
        corpus_name = os.path.basename(os.path.dirname(file_path))
        corpus = Dataset(corpus_name)
        if os.path.isdir(file_path):
            directory = file_path
            filenames = sorted(list(os.listdir(directory)))
            for filename in filenames:
                if filename.endswith(".txt"):
                    abs_path = os.path.join(directory, filename)
                    # print(filename)
                    Standoff.load_document(abs_path, filename, corpus)
        else:
            Standoff.load_document(file_path, corpus_name, corpus)
        return corpus

    @staticmethod
    def get_annotation_files(file_path: str) -> List[str]:
        assert file_path.endswith(".txt")
        base = file_path[:-4]
        annotation_files = ["%s.%s" % (base, ext) for ext in
                            Standoff.annotation_extensions]
        return [filename for filename in annotation_files if
                os.path.isfile(filename)]

    @staticmethod
    def load_entity(line: str, dataset: Dataset) -> None:
        pass

    @staticmethod
    def load_document(file_path: str, filename: str, dataset: Dataset) -> None:
        doc = dataset.create_document(filename)
        with codecs.open(file_path, "r", "utf-8") as f:
            text = f.read()
        for annotation_file in Standoff.get_annotation_files(file_path):
            with codecs.open(annotation_file, "r", "utf-8") as f:
                for line in f:
                    line = str(line.strip())
                    if line.startswith('E'):
                        Standoff.load_entity(line, dataset)
