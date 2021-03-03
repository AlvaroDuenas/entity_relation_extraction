import codecs
import os
import re
from typing import List, Dict
from transformers import BertTokenizer
from src.Entities import Dataset, Token, TokenSpan,Entity
from collections import OrderedDict
import spacy

nlp = spacy.load("en_core_web_sm")

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
    def load_entity(doc: Dict[int,Token], line: str, dataset: Dataset) -> None:
        ent_id, entities, text = tuple(line.split("\t"))
        text = text.replace("\xa0"," ")
        tokenized_text = [(token.text,token.idx,token.whitespace_,token.i) for token in nlp(text)]
        if ";" not in entities:
            ent_type, start, end = tuple(entities.split(" "))
            start = int(start)
            end = int(end)
            if ent_type in ["Title", "Paragraph"]:
                return
            tokens = []
            for i in range(start,end):
                if i in doc:
                    token = doc[i]
                    tokens.append(doc[i])
                    i+=len(token)
            check_text = TokenSpan(tokens).join()
            if text in check_text:
                return ent_id, dataset.create_entity(dataset.get_ent_type(ent_type),
                                      tokens, text)
                
    @staticmethod
    def load_relation(entities:Dict[str,Entity], line:str, 
                       dataset:Dataset):
        rel_id, rest = tuple(line.split("\t"))
        rel_type, head, tail = tuple(rest.split(" "))
        head_type, head_id = tuple(head.split(":"))
        tail_type, tail_id = tuple(tail.split(":"))
        if head_id in entities and tail_id in entities:
            relation = dataset.create_relation(dataset.get_rel_type(rel_type),
                                    entities[head_id],
                                    entities[tail_id])
            return rel_id, relation
        
    @staticmethod
    def load_document(file_path: str, filename: str, dataset: Dataset) -> None:
        with codecs.open(file_path, "r", "utf-8") as f:
            text = f.read().replace("\n", " ")
        #tokenized_doc = {token.idx :(token.text,token.idx,token.whitespace_,token.i, len(token.text_with_ws)) for token in nlp(text)}       
        tokens, encodings = Standoff.parse_tokens(text, dataset)
        aux = {token.idx:token for token in tokens.values()}
        entities = OrderedDict()
        relations = OrderedDict()
        for annotation_file in Standoff.get_annotation_files(file_path):
            with codecs.open(annotation_file, "r", "utf-8") as f:
                print(annotation_file)
                for line in f:
                    line = str(line.strip())
                    if line.startswith('T'):
                        aux2 = Standoff.load_entity(aux, line, dataset)
                        if aux2 is not None:
                            ent_id, ent = aux2
                            entities[ent_id] = ent
                    elif line.startswith('R'):
                        aux2 = Standoff.load_relation(entities, line, dataset)
                        if aux2 is not None:
                            rel_id, rel = aux2
                            relations[rel_id] = rel          
        doc = dataset.create_document(filename, entities,
                                      relations, tokens, encodings)
                        
    @staticmethod
    def parse_tokens(text: str, dataset: Dataset):
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased",
                                                  do_lower_case=True)
        doc_tokens = OrderedDict()
        doc_encoding = [tokenizer.convert_tokens_to_ids('[CLS]')]
        for token in nlp(text):
            token_encoding = tokenizer.encode(token.text,
                                              add_special_tokens=False)
            span_start, span_end = (len(doc_encoding),
                                    len(doc_encoding) +
                                    len(token_encoding))
            token = dataset.create_token(token.i, span_start, span_end,
                                         token.text, token.idx,
                                         token.whitespace_, 
                                         len(token.text_with_ws))
            doc_tokens[token.index] = token
            doc_encoding += token_encoding
        doc_encoding += [tokenizer.convert_tokens_to_ids('[SEP]')]
        return doc_tokens, doc_encoding
        
                

        
