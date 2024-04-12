import numpy as np
from tqdm.auto import tqdm

import evaluate
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup, AdamW

from utils import get_num_classes, initialize_networks, get_dataloader, get_tokenizer, load_model, save_model

import logging
logging.getLogger("imported_module").setLevel(logging.WARNING)

class LoTSolver(object):

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
        pos_count = torch.zeros((model.config.num_hidden_layers)).to(self.device)
        with torch.no_grad():
            for it, batch in enumerate(tqdm(dataloader)):
                inputs = {
                        "input_ids": batch['input_ids'].to(self.device),
                        "attention_mask": batch['attention_mask'].to(self.device),
                }
                if 'token_type_ids' in batch:
                    inputs['token_type_ids'] = batch['token_type_ids'].to(self.device)
                
                outputs = model(**inputs)
                
                # attention regularization
                all_pos_masks = self.encoder.all_pos_masks
                for i, mask in enumerate(all_pos_masks):
                    att_mask = inputs['attention_mask'].view(-1, mask.size(-1))
                    valid_mask = (mask * att_mask).sum(dim=-1) / att_mask.sum(dim=-1)
                    pos_count[i] += valid_mask.sum()

                nstep += len(batch['input_ids'])
                if preds is None:
                    preds = outputs.logits.detach().cpu().numpy()
                    out_label_ids = batch["label"].numpy()
                else:
                    preds = np.append(preds, outputs.logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, batch["label"].numpy(), axis=0)
        
        print('Routing results')
        routings = pos_count / nstep
        speedup = routings.mean().item()
        routings = routings.detach().cpu().numpy().tolist()
        routings = [ '%.2f' % elem for elem in routings ]
        print('position\t', routings, speedup)

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

        def top_p_sampling(probs, p=0.5):
            sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
            cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
            nucleus = cum_sum_probs < p
            nucleus[:, 0] = True
            
            for i in range(len(probs)):
                sorted_probs[i, indices[i, nucleus[i] == False]] = -1e4
            sorted_probs = (sorted_probs > 0).float()
            return sorted_probs


        # Load global validation set
        train_loader, test_loader = get_dataloader(dataset=self.dataset, train_bs=self.args.batch_size, test_bs=self.args.batch_size * 2)
        t_total = len(train_loader) * self.args.epochs

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if "skim" not in n],
                "lr": self.args.lr,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if "skim" in n],
                "lr": self.args.lr,
            },
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )
        
        if 'roberta' in self.args.model:
            self.encoder = self.model.roberta.encoder
        elif 'bert' in self.args.model:
            self.encoder = self.model.bert.encoder
        
        importance_maps = np.load(f'{self.args.task_grad}')
        importance_maps = torch.from_numpy(importance_maps).float().to(self.device)

        cur_step = 0
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
                index = batch['index'].to(self.device)
                if 'token_type_ids' in batch:
                    inputs['token_type_ids'] = batch['token_type_ids'].to(self.device)
                inputs['output_hidden_states'] = False


                token_dist = importance_maps[index].softmax(dim=-1)
                token_top_dist = top_p_sampling(token_dist, self.args.top_p).unsqueeze(-1)
                router_labels = torch.cat([1 - token_top_dist, token_top_dist], dim=-1)

                self.model.train()
                outputs = self.model(**inputs)
                loss = outputs.loss

                # attention regularization
                all_pos_logits = self.encoder.all_pos_logits
                pos_reg = 0.
                all_weights = 0
                for i, skim_logits in enumerate(all_pos_logits):
                    reg = -(router_labels[:, 1:] * F.log_softmax(skim_logits, dim=-1)).sum(dim=-1) * inputs['attention_mask'][:,1:]
                    pos_reg += reg.sum(dim=-1) / inputs['attention_mask'][:,1:].sum(dim=-1)
                    all_weights += 1
                pos_reg = pos_reg.mean() / all_weights

                loss = loss + pos_reg * self.args.reg_weight
                loss /= self.args.gradient_accumulation_step 
                loss.backward()

                if (it+1) % self.args.gradient_accumulation_step == 0 or (it+1) == len(train_loader):
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # gradient cliping

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                writer['loss']  += loss.mean().item()
                writer['step']  += 1
                cur_step += 1
                
                if (it+1) % self.args.logging_step == 0:
                    metric, test_acc = self.evaluate(self.model, test_loader)
                    print(f'Epoch ({epoch}, {it+1} step) test {metric} {test_acc}')
                    if test_acc > best_acc:
                        best_acc = test_acc
                        save_model(self.model, self.args.modeldir, f'{self.args.dataset}_{self.args.model}_{self.args.alg}_{test_acc}.pt')
                    self.model.train()
                    
            metric, test_acc = self.evaluate(self.model, test_loader)
            avg_loss = writer['loss']/writer['step']
            print(f'Epoch ({epoch}) avg loss {avg_loss} test {metric} {test_acc}')
            if test_acc > best_acc:
                best_acc = test_acc
                save_model(self.model, self.args.modeldir, f'{self.args.dataset}_{self.args.model}_{self.args.alg}_{test_acc}.pt')

