from src.Entities import EntityType, Dataset, Document
from sklearn.metrics import precision_recall_fscore_support as prfs
from typing import List, Tuple, Dict
from transformers import BertTokenizer
import torch

class Evaluator:
    def __init__(self, dataset: Dataset, text_encoder: BertTokenizer,
                 predictions_path: str,
                 epoch: int, dataset_label: str, rel_filter_threshold: float = 0.4,
                 no_overlapping: bool = False):
        self._text_encoder = text_encoder
        self._dataset = dataset
        self._rel_filter_threshold = rel_filter_threshold
        self._no_overlapping = no_overlapping

        self._epoch = epoch
        self._dataset_label = dataset_label

        self._predictions_path = predictions_path

        # relations
        self._gt_relations = []  # ground truth
        self._pred_relations = []  # prediction

        # entities
        self._gt_entities = []  # ground truth
        self._pred_entities = []  # prediction

        self._pseudo_entity_type = EntityType("Entity", 1)

        self._convert_gt(self._dataset.documents)

    def _convert_gt(self, docs: List[Document]):
        for doc in docs:
            gt_relations = doc.relations
            gt_entities = doc.entities

            # convert ground truth relations and entities for precision/recall/f1 evaluation
            sample_gt_entities = [entity.as_tuple() for entity in gt_entities]
            sample_gt_relations = [rel.as_tuple() for rel in gt_relations]

            # TODO overlapping

            self._gt_entities.append(sample_gt_entities)
            self._gt_relations.append(sample_gt_relations)

    def eval_batch(self, batch_entity_clf: torch.tensor, batch_rel_clf: torch.tensor,
                   batch_rels: torch.tensor, batch: dict):
        batch_size = batch_rel_clf.shape[0]
        rel_class_count = batch_rel_clf.shape[2]

        # get maximum activation (index of predicted entity type)
        batch_entity_types = batch_entity_clf.argmax(dim=-1)
        # apply entity sample mask
        batch_entity_types *= batch['entity_sample_masks'].long()

        batch_rel_clf = batch_rel_clf.view(batch_size, -1)

        # apply threshold to relations
        if self._rel_filter_threshold > 0:
            batch_rel_clf[batch_rel_clf < self._rel_filter_threshold] = 0

        for i in range(batch_size):
            # get model predictions for sample
            rel_clf = batch_rel_clf[i]
            entity_types = batch_entity_types[i]

            # get predicted relation labels and corresponding entity pairs
            rel_nonzero = rel_clf.nonzero().view(-1)
            rel_scores = rel_clf[rel_nonzero]

            # model does not predict None class (+1)
            rel_types = (rel_nonzero % rel_class_count) + 1
            rel_indices = rel_nonzero // rel_class_count

            rels = batch_rels[i][rel_indices]

            # get masks of entities in relation
            rel_entity_spans = batch['entity_spans'][i][rels].long()

            # get predicted entity types
            rel_entity_types = torch.zeros([rels.shape[0], 2])
            if rels.shape[0] != 0:
                rel_entity_types = torch.stack(
                    [entity_types[rels[j]] for j in range(rels.shape[0])])

            # convert predicted relations for evaluation
            sample_pred_relations = self._convert_pred_relations(rel_types, rel_entity_spans,
                                                                 rel_entity_types, rel_scores)

            # get entities that are not classified as 'None'
            valid_entity_indices = entity_types.nonzero().view(-1)
            valid_entity_types = entity_types[valid_entity_indices]
            valid_entity_spans = batch['entity_spans'][i][valid_entity_indices]
            valid_entity_scores = torch.gather(batch_entity_clf[i][valid_entity_indices], 1,
                                               valid_entity_types.unsqueeze(1)).view(-1)

            sample_pred_entities = self._convert_pred_entities(valid_entity_types, valid_entity_spans,
                                                               valid_entity_scores)

            # TODO overlapping

            self._pred_entities.append(sample_pred_entities)
            self._pred_relations.append(sample_pred_relations)

    def _convert_pred_entities(self, pred_types: torch.tensor, pred_spans: torch.tensor, pred_scores: torch.tensor):
        converted_preds = []

        for i in range(pred_types.shape[0]):
            label_idx = pred_types[i].item()
            entity_type = self._dataset.get_ent_type(label_idx)

            start, end = pred_spans[i].tolist()
            score = pred_scores[i].item()

            converted_pred = (start, end, entity_type, score)
            converted_preds.append(converted_pred)

        return converted_preds

    def _convert_pred_relations(self, pred_rel_types: torch.tensor, pred_entity_spans: torch.tensor,
                                pred_entity_types: torch.tensor, pred_scores: torch.tensor):
        converted_rels = []
        check = set()

        for i in range(pred_rel_types.shape[0]):
            label_idx = pred_rel_types[i].item()
            pred_rel_type = self._dataset.get_rel_type(label_idx)
            pred_head_type_idx, pred_tail_type_idx = pred_entity_types[i][0].item(
            ), pred_entity_types[i][1].item()
            pred_head_type = self._dataset.get_ent_type(pred_head_type_idx)
            pred_tail_type = self._dataset.get_ent_type(pred_tail_type_idx)
            score = pred_scores[i].item()

            spans = pred_entity_spans[i]
            head_start, head_end = spans[0].tolist()
            tail_start, tail_end = spans[1].tolist()

            converted_rel = ((head_start, head_end, pred_head_type),
                             (tail_start, tail_end, pred_tail_type), pred_rel_type)
            converted_rel = self._adjust_rel(converted_rel)

            if converted_rel not in check:
                check.add(converted_rel)
                converted_rels.append(tuple(list(converted_rel) + [score]))

        return converted_rels

    def compute_scores(self):
        print("Evaluation")

        print("")
        print("--- Entities (named entity recognition (NER)) ---")
        print("An entity is considered correct if the entity type and span is predicted correctly")
        print("")
        gt, pred = self._convert_by_setting(
            self._gt_entities, self._pred_entities, include_entity_types=True)
        ner_eval = self._score(gt, pred, print_results=True)

        print("")
        print("--- Relations ---")
        print("")
        print("Without named entity classification (NEC)")
        print("A relation is considered correct if the relation type and the spans of the two "
              "related entities are predicted correctly (entity type is not considered)")
        print("")
        gt, pred = self._convert_by_setting(
            self._gt_relations, self._pred_relations, include_entity_types=False)
        rel_eval = self._score(gt, pred, print_results=True)

        print("")
        print("With named entity classification (NEC)")
        print("A relation is considered correct if the relation type and the two "
              "related entities are predicted correctly (in span and entity type)")
        print("")
        gt, pred = self._convert_by_setting(
            self._gt_relations, self._pred_relations, include_entity_types=True)
        rel_nec_eval = self._score(gt, pred, print_results=True)

        return ner_eval, rel_eval, rel_nec_eval

    def _score(self, gt: List[List[Tuple]], pred: List[List[Tuple]], print_results: bool = False):
        assert len(gt) == len(pred)

        gt_flat = []
        pred_flat = []
        types = set()

        for (sample_gt, sample_pred) in zip(gt, pred):
            union = set()
            union.update(sample_gt)
            union.update(sample_pred)

            for s in union:
                if s in sample_gt:
                    t = s[2]
                    gt_flat.append(t.index)
                    types.add(t)
                else:
                    gt_flat.append(0)

                if s in sample_pred:
                    t = s[2]
                    pred_flat.append(t.index)
                    types.add(t)
                else:
                    pred_flat.append(0)

        metrics = self._compute_metrics(
            gt_flat, pred_flat, types, print_results)
        return metrics

    def _compute_metrics(self, gt_all, pred_all, types, print_results: bool = True):
        labels = [t.index for t in types]
        per_type = prfs(gt_all, pred_all, labels=labels, average=None)
        micro = prfs(gt_all, pred_all, labels=labels, average='micro')[:-1]
        macro = prfs(gt_all, pred_all, labels=labels, average='macro')[:-1]
        total_support = sum(per_type[-1])

        if print_results:
            self._print_results(per_type, list(
                micro) + [total_support], list(macro) + [total_support], types)

        return [m * 100 for m in micro + macro]
    
    def _print_results(self, per_type: List, micro: List, macro: List, types: List):
        columns = ('type', 'precision', 'recall', 'f1-score', 'support')

        row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
        results = [row_fmt % columns, '\n']

        metrics_per_type = []
        for i, t in enumerate(types):
            metrics = []
            for j in range(len(per_type)):
                metrics.append(per_type[j][i])
            metrics_per_type.append(metrics)

        for m, t in zip(metrics_per_type, types):
            results.append(row_fmt % self._get_row(m, t.short_name))
            results.append('\n')

        results.append('\n')

        # micro
        results.append(row_fmt % self._get_row(micro, 'micro'))
        results.append('\n')

        # macro
        results.append(row_fmt % self._get_row(macro, 'macro'))

        results_str = ''.join(results)
        print(results_str)

    def _convert_by_setting(self, gt: List[List[Tuple]], pred: List[List[Tuple]],
                            include_entity_types: bool = True, include_score: bool = False):
        assert len(gt) == len(pred)

        # either include or remove entity types based on setting
        def convert(t):
            if not include_entity_types:
                # remove entity type and score for evaluation
                if type(t[0]) == int:  # entity
                    c = [t[0], t[1], self._pseudo_entity_type]
                else:  # relation
                    c = [(t[0][0], t[0][1], self._pseudo_entity_type),
                         (t[1][0], t[1][1], self._pseudo_entity_type), t[2]]
            else:
                c = list(t[:3])

            if include_score and len(t) > 3:
                # include prediction scores
                c.append(t[3])

            return tuple(c)

        converted_gt, converted_pred = [], []

        for sample_gt, sample_pred in zip(gt, pred):
            converted_gt.append([convert(t) for t in sample_gt])
            converted_pred.append([convert(t) for t in sample_pred])

        return converted_gt, converted_pred
