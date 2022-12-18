import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import numpy as np
from transformers import BertTokenizer
import pandas as pd
from smart_open import open as smart_open
import io
import os
from tqdm import tqdm
from dataloader import get_dataloader, check_cache, prepare_features, process_data, prepare_inputs


class StudentModel(nn.Module):
    def __init__(self, config):
        super(StudentModel, self).__init__()
        

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.rnn = nn.LSTM(input_size=config.embedding_size, hidden_size=config.hidden_size, batch_first=True, num_layers = config.num_layers, bidirectional=True)
        self.fc = nn.Sequential(
            #(batch_size, hidden_size)
            nn.Linear(2 * config.hidden_size, config.fc_size),
            nn.ReLU(True),
            nn.Dropout(config.dropout),
            #(batch_size, hidden_size)
            nn.Linear(config.fc_size, config.n_classes),
            #(batch_size, n_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        #input_ids.shape = (batch_size, seq_len)

        #embedding.shape = (batch_size, seq_len, hidden_size)
        embedding = self.embedding(input_ids)

        #rnn.shape = (batch_size, seq_len, hidden_size)
        rnn, _ = self.rnn(embedding)

        #last.shape = (batch_size, hidden_size)
        #print(rnn)
        #print(rnn.size())
        last = rnn[:, -1, :]
        #packed = torch.nn.utils.rnn.pack_padded_sequence(a, length, batch_first=True, enforce_sorted = False)
        #.shape = (batch_size, n_classes)
        return self.fc(last)


# train




def learn(config, teacher_model, datasets, tokenizer):
    hard_loss = nn.CrossEntropyLoss()
    soft_loss = nn.KLDivLoss(reduction='batchmean')

    student_model = StudentModel(config).to('cuda')
    student_optimizer = optim.AdamW(student_model.parameters(), lr=config.lr, weight_decay = config.weight_decay)

    current_path = os.path.dirname(__file__)
    save_student_path = os.path.join(current_path, './saved_model/student.pt')

    if os.path.exists(save_student_path):
        print('load student state dict')
        student_state_dict = torch.load(save_student_path)
        student_model.load_state_dict(student_state_dict)
    temp = config.temp
    alpha = config.alpha

    teacher_model.eval()
    
    train_dataloader = get_dataloader(config, datasets['train'], 'train')

    best_acc = 0
    early_stop = config.early_stop
    for epoch in range(config.epochs):
        losses = 0
        student_model.train()
        with tqdm(total=len(train_dataloader)) as p:
            for step, batch in enumerate(train_dataloader):
                inputs, labels = prepare_inputs(batch)
                #print('inputs', inputs['input_ids'].size())
                '''
                print('inputs-----ids-------')
                print(inputs['input_ids'])
                print('inputs-----token type id-------')
                print(inputs['token_type_ids'])
                print('inputs-----attention mask-------')
                print(inputs['attention_mask'])                
                print('labels------------')
                print(labels)
                '''
                with torch.no_grad():
                    teacher_prediciton = teacher_model(inputs, labels)
                '''
                print('teacher', next(teacher_model.parameters()).is_cuda)

                print('std', next(student_model.parameters()).is_cuda)
                '''
                student_prediction = student_model(inputs['input_ids'], inputs['attention_mask'])

                student_hard_loss = hard_loss(student_prediction, labels)
                student_soft_loss = soft_loss(
                    F.log_softmax(student_prediction/config.temp, dim=-1),
                    F.softmax(teacher_prediciton/config.temp, dim=-1))
                
                student_loss = config.alpha*student_soft_loss + (1.0 - config.alpha)*student_hard_loss

                student_optimizer.zero_grad()
                student_loss.backward()
                # print(student_model.embedding.weight.grad);input()
                student_optimizer.step()

                p.set_postfix({'loss': student_loss.item()})
                p.update()
                losses += student_loss.item()
        print('epoch', epoch, '| train losses', losses/len(train_dataloader))

        acc = _eval(split = 'validation')
        if acc/len(datasets['validation']) > best_acc:
            torch.save(student_model.state_dict(), save_student_path)
            early_stop = config.early_stop
        else:
            early_stop -= 1
            if early_stop == 0:

                eval(split = 'test')
                return 'early stop ends'
    eval(split = 'test')
    return 'end of epochs'


def _eval(split = 'validation'):
    student_model.eval()
    dataloader = get_dataloader(config, datasets[split], split)
    acc = 0
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch)
        logits = student_model(inputs['input_ids'], inputs['attention_mask'])
        acc += (logits.argmax(1) == labels).float().sum().item()
        #acc += tem.item()
    

    print(f'{split} acc:', acc/len(datasets[split]), f'|dataset split {split} size:', len(datasets[split]))
    return acc
#torch.save(student_model.state_dict(), save_student_path)