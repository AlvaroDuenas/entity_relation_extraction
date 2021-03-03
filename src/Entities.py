from collections import OrderedDict
from typing import List, Dict
import src.procesamiento
from torch.utils.data import Dataset as TorchDataset
class Token:
    def __init__(self, tok_id: int, index: int,
                 span_start: int, span_end: int, text: str,
                 idx: int, ws: str, length: int):
        self._tok_id = tok_id
        self._index = index
        self._span_start = span_start
        self._span_end = span_end
        self._text = text
        self._ws = ws
        self._idx = idx
        self._length = length

    @property
    def index(self):
        return self._index

    @property
    def ws(self):
        return self._ws

    @property
    def idx(self):
        return self._idx

    @property
    def tok_id(self):
        return self._tok_id

    @property
    def span_start(self):
        return self._span_start

    @property
    def span_end(self):
        return self._span_end

    @property
    def span(self):
        return self._span_start, self._span_end

    @property
    def text(self):
        return self._text

    def __eq__(self, other):
        if isinstance(other, Token):
            return self._tid == other._tid
        return False

    def __hash__(self):
        return hash(self._tid)

    def __str__(self):
        return self._text

    def __repr__(self):
        return self._text

    def __len__(self):
        return self._length


class TokenSpan:
    def __init__(self, tokens: List[Token]):
        self._tokens = tokens

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    def __getitem__(self, s):
        if isinstance(s, slice):
            return TokenSpan(self._tokens[s.start:s.stop:s.step])
        else:
            return self._tokens[s]

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)

    def join(self):
        return "".join([token.text+token.ws for token in self._tokens])


class EntityType:
    def __init__(self, ent_type: str, index: int):
        self._ent_type = ent_type
        self._index = index

        @property
        def ent_type(self):
            return self._ent_type

        @property
        def index(self):
            return self._index

        def __int__(self):
            return self._index

        def __eq__(self, other):
            if isinstance(other, EntityType):
                return self._ent_type == other._ent_type
            return False

        def __hash__(self):
            return hash(self._ent_type)


class RelationType:
    def __init__(self, rel_type: str, index: int):
        self._rel_type = rel_type
        self._index = index

        @property
        def rel_type(self):
            return self._rel_type

        @property
        def index(self):
            return self._index

        def __int__(self):
            return self._index

    def __eq__(self, other):
        if isinstance(other, RelationType):
            return self._rel_type == other._rel_type
        return False

    def __hash__(self):
        return hash(self._rel_type)


class Entity:
    def __init__(self, ent_id: int, entity_type: EntityType,
                 tokens: List[Token], text: str):
        self._ent_id = ent_id
        self._entity_type = entity_type
        self._tokens = tokens
        self._text = text

    @property
    def entity_type(self):
        return self._entity_type

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    def as_tuple(self):
        return (self.tokens.span_start,
                self.tokens.span_end, self._entity_type)

    @property
    def text(self):
        return self._text

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self._ent_id == other._ent_id
        return False

    def __hash__(self):
        return hash(self._ent_id)

    def __str__(self):
        return self._text


class Relation:
    def __init__(self, rel_id: int, rel_type: RelationType,
                 head_entity: Entity,
                 tail_entity: Entity):
        self._rel_id = rel_id
        self._rel_type = rel_type
        self._head_entity = head_entity
        self._tail_entity = tail_entity

    def as_tuple(self):
        return (self._head_entity.as_tuple(),
                self.tail_entity.as_tuple(), self._rel_type.rel_type)

    @property
    def rel_type(self):
        return self._rel_type

    @property
    def head_entity(self):
        return self._head_entity

    @property
    def tail_entity(self):
        return self._tail_entity

    def __eq__(self, other):
        if isinstance(other, Relation):
            return self._rel_id == other._rel_id
        return False

    def __hash__(self):
        return hash(self._rel_id)


class Document:
    def __init__(self, file_name: str, doc_id: int, entities: Dict[str, Entity],
                 relations: Dict[str, Relation], tokens: Dict[int, Token],
                 encoding: List[int]):
        self._name = file_name
        self._doc_id = doc_id
        self._relations = relations
        self._entities = entities
        self._tokens = tokens
        self._encoding = encoding

    def __eq__(self, other):
        if isinstance(other, Document):
            return self._doc_id == other._doc_id
        return False

    def __hash__(self):
        return hash(self._doc_id)


class Dataset(TorchDataset):
    def __init__(self, corpus_name: str, mode: bool = True,
                 neg_entity_count:int=100,neg_rel_count:int=100,
                 max_span_size:int=10):
        self._id = corpus_name
        self._documents = OrderedDict()
        self._relations = OrderedDict()
        self._entities = OrderedDict()
        self._ent_types = OrderedDict()
        self._rel_types = OrderedDict()
        
        self._rel_types["None"] = RelationType("None", 0)
        
        self._mode = mode
        self._neg_entity_count = neg_entity_count
        self._neg_rel_count = neg_rel_count
        self._max_span_size = max_span_size

        self._doc_id = 0
        self._rel_id = 0
        self._ent_id = 0
        self._tok_id = 0
        self._rel_type_id = 1
        self._ent_type_id = 0

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value

    def create_document(self, filename: str, entities: Dict[str, Entity],
                        relations: Dict[str, Relation], tokens: Dict[int, Token],
                        encoding: List[int]) -> Document:
        doc = Document(filename, self._doc_id, entities,
                       relations, tokens, encoding)
        self._documents[self._doc_id] = doc
        self._doc_id += 1
        return doc

    def create_entity(self, ent_type: EntityType, tokens: List[Token],
                      text: str) -> Entity:
        entity = Entity(self._ent_id, ent_type,
                        tokens, text)
        self._entities[self._ent_id] = entity
        self._ent_id += 1
        return entity

    def create_relation(self, rel_type: RelationType,
                        head_entity: Entity,
                        tail_entity: Entity) -> Relation:
        relation = Relation(self._rel_id, rel_type,
                            head_entity, tail_entity)
        self._relations[self._rel_id] = relation
        self._rel_id += 1
        return relation

    def create_token(self, i: int, span_start: int, span_end: int,
                     text: str, idx: int, ws: str, length: str) -> Token:
        token = Token(self._tok_id, i, span_start, span_end,
                      text, idx, ws, length)
        self._tok_id += 1
        return token

    def get_rel_type(self, rel_type: str) -> RelationType:
        if rel_type not in self._rel_types:
            self._rel_types[rel_type] = RelationType(
                rel_type, self._rel_type_id)
            self._rel_type_id += 1
        return self._rel_types[rel_type]

    def get_ent_type(self, ent_type: str) -> EntityType:
        if ent_type not in self._ent_types:
            self._ent_types[ent_type] = EntityType(
                ent_type, self._ent_type_id)
            self._ent_type_id += 1
        return self._ent_types[ent_type]

    def __len__(self):
        return len(self._documents)

    def __getitem__(self, index: int):
        doc = self._documents[index]

        if self._mode:
            return procesamiento.create_train_sample(doc, self._neg_entity_count,
                                                self._neg_rel_count,
                                                self._max_span_size,
                                                len(self._rel_types))
        else:
            return procesamiento.create_eval_sample(doc, self._max_span_size)
