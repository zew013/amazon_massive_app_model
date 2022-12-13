from itertools import chain
import os, sys, pdb
import numpy as np
import random
import torch

import math

from tqdm import tqdm as progress_bar

from utils import set_seed, setup_gpus, check_directories, AdamW_LLRD
from dataloader import get_dataloader, check_cache, prepare_features, process_data, prepare_inputs
from load import load_data, load_tokenizer
from arguments import params
from model import IntentModel, SupConModel, CustomModel
from torch import nn

import transformers

device='cuda'

def baseline_train(args, model, datasets, tokenizer):
  
  
    # print all the named parameters in the model
    # for name, param in model.named_parameters():
    #   print(name, param.shape)
    
    
  
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets['train'], 'train')

    
    
    model.optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
                                      
    model.scheduler = transformers.get_scheduler(args.scheduler_type,
                                                 model.optimizer,
                                                 num_warmup_steps=args.warmup_steps,
                                                 num_training_steps=int(args.n_epochs*len(train_dataloader)/(1 - args.end_lr_ratio)))
                                                                   
    
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        with progress_bar(total=len(train_dataloader)) as p:
          for step, batch in enumerate(train_dataloader):
              inputs, labels, target_text = prepare_inputs(batch, use_text=True)
              
              # print(inputs, target_text)
              
              logits = model(inputs, labels)
              
              # print the max logits and the corresponding label
              
              # print(logits[0], labels[0])
              loss = criterion(logits, labels)
              loss.backward()

              model.optimizer.step()  # backprop to update the weights
              model.zero_grad()

              model.scheduler.step()  # Update learning rate schedule
              
              p.set_postfix({'loss': loss.item(), 'lr': model.scheduler.get_last_lr()[0]})
              p.update()

              
              losses += loss.item()
         
          # model.scheduler.step()  # Update learning rate schedule 
        
    
        run_eval(args, model, datasets, tokenizer)
        print('epoch', epoch_count, '| losses:', losses/len(train_dataloader))
  
def custom_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets['train'], 'train')

    if args.lldr:
      optimizer = AdamW_LLRD(args, model)
      print('adamw llrd')
    else:
      optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
                                
    scheduler = transformers.get_scheduler(args.scheduler_type,
                                                 optimizer,
                                                 num_warmup_steps=args.warmup_steps,
                                                 num_training_steps=args.n_epochs*len(train_dataloader))

    
    #model.scheduler = transformers.get_constant_schedule(model.optimizer)
    
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        with progress_bar(total=len(train_dataloader)) as p:
            for step, batch in enumerate(train_dataloader):
                inputs, labels, target_text = prepare_inputs(batch, use_text=True)
              
                # print(inputs, target_text)
              
                logits = model(inputs, labels)
              
                # print the max logits and the corresponding label
              
                # print(logits[0], labels[0])
                loss = criterion(logits, labels)
                loss.backward()

                optimizer.step()  # backprop to update the weights
                optimizer.zero_grad()

                #model.scheduler.step()  # Update learning rate schedule
                    
                p.set_postfix({'loss': loss.item()})
                p.update()
                losses += loss.item()

            scheduler.step()  # Update learning rate schedule
        
        run_eval(args, model, datasets, tokenizer)
        print('epoch', epoch_count, '| losses:', losses/len(train_dataloader))
        

def run_eval(args, model, datasets, tokenizer, split='validation'):
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)

    acc = 0
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch)
        logits = model(inputs, labels)
        tem = (logits.argmax(1) == labels).float().sum()
        acc += tem.item()
  
    print(f'{split} acc:', acc/len(datasets[split]), f'|dataset split {split} size:', len(datasets[split]))

def supcon_train(args, model, datasets, tokenizer):
    from loss import SupConLoss
    criterion = SupConLoss(temperature=args.temperature)
  
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets['train'], 'train')
    
    
    model.optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
                                      
    model.scheduler = transformers.get_scheduler(args.scheduler_type,
                                                 model.optimizer,
                                                 num_warmup_steps=args.warmup_steps,
                                                 num_training_steps=args.n_epochs*len(train_dataloader))
    
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        with progress_bar(total=len(train_dataloader)) as p:
          for step, batch in enumerate(train_dataloader):
            
              # make two copies of the inputs as the two views; dropout will do the trick of augmenting the data
              inputs, labels, target_text  = prepare_inputs(batch, use_text=True)
              
              inputs = {k: torch.cat((inputs[k],) * 2, dim=0) for k in inputs}
              
              features = model(inputs, labels)
              
              
              f1, f2 = torch.split(features, (len(labels),) * 2, dim=0)
              
              features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
              
              
              if args.task == 'supcon':
                loss = criterion(features, labels)
              else:
                loss = criterion(features)
              loss.backward()

              model.optimizer.step()  # backprop to update the weights
              model.zero_grad()

              model.scheduler.step()  # Update learning rate schedule
              
              p.set_postfix({'loss': loss.item(), 'lr': model.scheduler.get_last_lr()[0]})
              p.update()

              
              losses += loss.item()
          
          print('epoch', epoch_count, '| losses:', losses/len(train_dataloader))
         
        
    # # change to fine-tuning mode; @965 we don't need to finetune...
    # criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # model.__call__ = model.classify
    
    # model.optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate * 10, eps=args.adam_epsilon)
                                                
    
    # # freeze the encoder
    # for param in model.encoder.parameters():
    #   if param.requires_grad:
    #     param.requires_grad = False
    
    # model.classifier.requires_grad = True
     
    # for epoch_count in range(args.n_epochs):
    #   losses = 0
    #   model.train()

    #   with progress_bar(total=len(train_dataloader)) as p:
    #     for step, batch in enumerate(train_dataloader):
    #       inputs, labels  = prepare_inputs(batch, model)
          
    #       logits = model(inputs, labels)
          
    #       loss = criterion(logits, labels)
    #       loss.backward()

    #       model.optimizer.step()  # backprop to update the weights
    #       model.zero_grad()

    #       model.scheduler.step()  # Update learning rate schedule
          
    #       p.set_postfix({'loss': loss.item(), 'lr': model.scheduler.get_last_lr()[0]})
    #       p.update()

          
    #       losses += loss.item()
    
        # run_eval(args, model, datasets, tokenizer)
        # print('epoch', epoch_count, '| losses:', losses/len(train_dataloader))


def do_umap(args, model, datasets, split='validation', b_size=256, classes=range(10)):
  dataloader = get_dataloader(args, datasets[split], split, b_size, classes=classes)
  model.eval()
  
  embeddings = []
  all_labels = []
  
  for batch in dataloader:

    batch = next(iter(dataloader))
    inputs, labels  = prepare_inputs(batch)
  
    # select labels 1-10 only
    model(inputs, labels)
    embeddings.append(model.embeddings.detach().cpu().numpy())
    all_labels.append(labels.cpu().numpy())
  
  model.to(torch.device("cpu"))
  embeddings = np.concatenate(embeddings, axis=0)
  all_labels = np.concatenate(all_labels, axis=0)
  
  
  mapper = umap.UMAP().fit(embeddings)
  res = umap.plot.points(mapper, labels=all_labels)
  umap.plot.show(res)
  return res 


def main(args):
  args = setup_gpus(args)
  args = check_directories(args)
  set_seed(args)

  cache_results, already_exist = check_cache(args)
  tokenizer = load_tokenizer(args)

  if already_exist:
    features = cache_results
  else:
    data = load_data()
    features = prepare_features(args, data, tokenizer, cache_results)
  datasets = process_data(args, features, tokenizer)
  for k,v in datasets.items():
    print(k, len(v))
  
  
  if args.task == 'baseline':
    model = IntentModel(args, tokenizer, target_size=60).to(device)
    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    baseline_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')
  elif args.task == 'custom': # you can have multiple custom task for different techniques
    model = CustomModel(args, tokenizer, target_size=60).to(device)
    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    custom_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')
    
  elif args.task == 'supcon':
    model = SupConModel(args, tokenizer, target_size=60).to(device)
    supcon_train(args, model, datasets, tokenizer)
  elif args.task == 'simclr':
    model = SupConModel(args, tokenizer, target_size=60).to(device)
    supcon_train(args, model, datasets, tokenizer)
    do_umap(args, model, datasets, split='test')
  return model, datasets, tokenizer
   


  

if __name__ == "__main__":
  args = params()
  
  main(args)
  