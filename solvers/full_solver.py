
import numpy as np
from tqdm.auto import tqdm

import evaluate

import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup, AdamW

from utils import get_num_classes, initialize_networks, get_dataloader, get_tokenizer, load_model, save_model

import logging
logging.getLogger("imported_module").setLevel(logging.WARNING)

from torch.cuda.amp import GradScaler 

class FullSolver(object):

    def __init__(self, args, dataset):
        """ Initialize configurations. """
        self.args = args
        self.dataset = dataset
        self.num_class = get_num_classes(args.dataset)
        self.tokenizer = get_tokenizer(args.model, max_length=args.max_seq_len)

        # Load training networks
        self.model = initialize_networks(alg=args.alg, dataset=args.dataset, model=args.model)
        print(self.model)

        # Optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.device = torch.device('cuda')
    
    def evaluate(self, model, dataloader):
        
        avg_acc = 0.
        nstep = 0
                
        results = {}
        preds = None
        out_label_ids = None
        
        model.eval()
        with torch.no_grad():
            for it, batch in enumerate(tqdm(dataloader)):
                inputs = {
                        "input_ids": batch['input_ids'].to(self.device),
                        "attention_mask": batch['attention_mask'].to(self.device),
                }
                if 'token_type_ids' in batch:
                    inputs['token_type_ids'] = batch['token_type_ids'].to(self.device)
                
                outputs = model(**inputs)
                
                nstep += len(batch['input_ids'])
                if preds is None:
                    preds = outputs.logits.detach().cpu().numpy()
                    out_label_ids = batch["label"].numpy()
                else:
                    preds = np.append(preds, outputs.logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, batch["label"].numpy(), axis=0)

        if self.args.dataset in ['sst2', 'mrpc', 'imdb', 'mnli', 'qnli', 'stsb']: # classification
            if self.args.dataset == 'stsb':
                preds = preds[:, 0]
            else:
                preds = np.argmax(preds, axis=1)
            if self.args.dataset == 'imdb':
                accuracy = evaluate.load("glue", 'sst2')
            else:
                accuracy = evaluate.load("glue", self.args.dataset)
            results = accuracy.compute(predictions=preds, references=out_label_ids)
        task2measure = {'sst2': 'accuracy', 'mrpc': 'f1', 'imdb': 'accuracy', 'mnli': 'accuracy', 'qnli': 'accuracy', 'stsb': 'pearson'}
        return task2measure[self.args.dataset], results[task2measure[self.args.dataset]]
                
                

    def run(self):
        """ Start federated learning scenario """
        # Load global validation set
        train_loader, test_loader = get_dataloader(dataset=self.dataset, train_bs=self.args.batch_size, test_bs=self.args.batch_size)
        t_total = len(train_loader) * self.args.epochs

        optimizer = AdamW(self.model.parameters(), lr=self.args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )

        best_acc = -1
        self.model = self.model.to(self.device)
        for epoch in range(self.args.epochs):
            writer = {'loss': 0., 'acc': 0., 'step': 0}
            self.model.train()
            for it, batch in enumerate(tqdm(train_loader)):
                
                inputs = {
                    "input_ids": batch['input_ids'].to(self.device),
                    "attention_mask": batch['attention_mask'].to(self.device),
                    "labels": batch['label'].to(self.device),
                }
                if 'token_type_ids' in batch:
                    inputs['token_type_ids'] = batch['token_type_ids'].to(self.device)
                outputs = self.model(**inputs)
                loss = outputs.loss

                # Model Updates
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                writer['loss']  += loss.mean().item()
                writer['step']  += 1
                
                if (it+1) % self.args.logging_step == 0:
                    metric, test_acc = self.evaluate(self.model, test_loader)
                    print(f'Epoch ({epoch}, {it+1} step) test {metric} {test_acc}')
                    if test_acc > best_acc:
                        best_acc = test_acc
                        save_model(self.model, self.args.modeldir, f'{self.args.dataset}_{self.args.model.split("/")[-1]}_{self.args.alg}_{test_acc}.pt')
                    self.model.train()
                    
            metric, test_acc = self.evaluate(self.model, test_loader)
            avg_loss = writer['loss'] / writer['step']
            print(f'Epoch ({epoch}) avg loss {avg_loss} test {metric} {test_acc}')
            if test_acc > best_acc:
                best_acc = test_acc
                save_model(self.model, self.args.modeldir, f'{self.args.dataset}_{self.args.model.split("/")[-1]}_{self.args.alg}_{test_acc}.pt')

