from transformers import BertTokenizer, BertConfig, AdamW
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from src import procesamiento
from src.Readers import Standoff
from src.Entities import Dataset
from src.Model import Model, Loss
from src.Evaluator import Evaluator
import transformers
from tqdm import tqdm


class App:
    def __init__(self):
        self._tokenizer = BertTokenizer.from_pretrained("bert-base-cased",
                                                        do_lower_case=True)
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        self._gpu_count = torch.cuda.device_count()

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    def train(self, train_path: str, dev_path: str,
              train_batch_size: int = 100, epochs: int = 10,
              max_pairs: int = 1000, prop_drop: float = 0.1,
              size_embedding: int = 25, max_grad_norm: float = 1.0):
        train_dataset = Standoff.load(train_path)
        dev_dataset = Standoff.load(dev_path)

        train_sample_count = len(train_dataset)
        updates_epoch = train_sample_count // train_batch_size
        updates_total = updates_epoch * epochs

        config = BertConfig.from_pretrained("bert-base-cased")

        model = Model.from_pretrained("bert-base-cased",
                                      config=config,
                                      cls_token=self._tokenizer.convert_tokens_to_ids(
                                          '[CLS]'),
                                      relation_types=train_dataset.num_rel_types(),
                                      entity_types=train_dataset.num_ent_types(),
                                      max_pairs=max_pairs,
                                      prop_drop=prop_drop,
                                      size_embedding=size_embedding)
        model.to(self._device)
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr,
                          weight_decay=args.weight_decay, correct_bias=False)
        # create scheduler
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=args.lr_warmup * updates_total,
                                                                 num_training_steps=updates_total)
        # create loss function
        rel_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        compute_loss = Loss(rel_criterion, entity_criterion, model,
                            optimizer, scheduler, max_grad_norm)
        for epoch in range(epochs):
            self._train_epoch(model, compute_loss, optimizer,
                              train_dataset, updates_epoch, epoch, train_batch_size)
            self.eval(model, validation_dataset, epoch + 1, updates_epoch)
                
    def _train_epoch(self, model: torch.nn.Module, compute_loss: Loss,
                     optimizer: Optimizer, dataset: Dataset,
                     updates_epoch: int, epoch: int, 
                     train_batch_size:int, sampling_processes:int):
        # create data loader
        dataset.mode(True)
        data_loader = DataLoader(dataset, batch_size=train_batch_size, 
                                 shuffle=True, drop_last=True,
                                 num_workers=sampling_processes, 
                                 collate_fn=procesamiento.collate_fn_padding)

        model.zero_grad()

        iteration = 0
        total = len(dataset) // train_batch_size
        for batch in tqdm(data_loader, total=total, desc='Train epoch %s' % epoch):
            model.train()
            batch = util.to_device(batch, self._device)

            # forward step
            entity_logits, rel_logits = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                                              entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                                              relations=batch['rels'], rel_masks=batch['rel_masks'])

            # compute loss and optimize parameters
            batch_loss = compute_loss.compute(entity_logits=entity_logits, rel_logits=rel_logits,
                                              rel_types=batch['rel_types'], entity_types=batch['entity_types'],
                                              entity_sample_masks=batch['entity_sample_masks'],
                                              rel_sample_masks=batch['rel_sample_masks'])

            iteration += 1
            global_iteration = epoch * updates_epoch + iteration
        return iteration
    def eval(self, model: torch.nn.Module,
             dataset: Dataset, epoch: int = 0, updates_epoch: int = 0,
             iteration: int = 0, eval_batch_size: int = 1,
             sampling_processes: int = 4):
        evaluator = Evaluator(dataset, self._tokenizer,
                              self._predictions_path,
                              epoch, dataset.label)
        data_loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=sampling_processes, collate_fn=procesamiento.collate_fn_padding)

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(len(dataset) /
                              self.args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                result = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                               entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                               entity_spans=batch['entity_spans'], entity_sample_masks=batch['entity_sample_masks'],
                               evaluate=True)
                entity_clf, rel_clf, rels = result

                # evaluate batch
                evaluator.eval_batch(entity_clf, rel_clf, rels, batch)

        global_iteration = epoch * updates_epoch + iteration
        ner_eval, rel_eval, rel_nec_eval = evaluator.compute_scores()
        
        #TODO store predictions and samples
