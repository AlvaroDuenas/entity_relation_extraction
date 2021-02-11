from collections import OrderedDict

class Document:
    def __init__(self, file_name: str, doc_id: int):
        self._name = file_name
        self._doc_id = doc_id
        self._relations = OrderedDict()
        self._entities = OrderedDict()


class Dataset:
    def __init__(self, corpus_name: str):
        self._id = corpus_name
        self._documents = OrderedDict()
        self._relations = OrderedDict()
        self._entities = OrderedDict()
        self._entity_types = OrderedDict()
        self._relation_types = OrderedDict()
        self._did = 0
        self._rid = 0
        self._eid = 0

    def create_document(self, filename: str) -> Document:
        doc = Document(filename, self._did)
        self._documents[self._did] = doc
        self._did += 1
        return doc


