
import pandas as pd
import numpy as np
import json
import time
import math
import re
import os
import torch
from transformers import  AdamW
from transformers import BertTokenizer, BertConfig, BertModel, BertPreTrainedModel, BertForTokenClassification
from transformers import XLNetTokenizer, XLNetConfig, XLNetModel, XLNetPreTrainedModel, XLNetForTokenClassification
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup
import ast
from ckiptagger import WS, POS, NER
import sys
# from ckiptagger import data_utils
# data_utils.download_data_gdown("./")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ws = WS("./data", disable_cuda=False)
pos = POS("./data", disable_cuda=False)
ner = NER("./data", disable_cuda=False)

class TestingDataset(Dataset):
    def __init__(self, MAX_LEN, tokenizer, data):
        self.data = [open(data, "r").read()]
        self.len = len(self.data)
        self.tokenizer = tokenizer  

    def __len__(self):
        return self.len
    
    def clean_text(self,text):
        text = text.replace('\r','')
        text = text.replace('\n','')
        return text
    
    def __getitem__(self, idx):
        MAX_LEN = 600
        content = self.data[idx]
        text = self.clean_text(content)
        #print('content',content)
        inputs = self.tokenizer.encode_plus(text=text, max_length=MAX_LEN, return_tensors='pt', 
                                            pad_to_max_length = True, 
                                            return_token_type_ids = True,
                                            return_attention_mask=True)
        input_ids = inputs['input_ids'].squeeze(0)
        segments_tensor = inputs['token_type_ids'].squeeze(0)
        masks_tensor = inputs['attention_mask'].squeeze(0)
        return input_ids
        
class XLNetForTokenClassification(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 5 
        self.transformer = XLNetModel(config)
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=True,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
           
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[1:]  # Keep mems, hidden states, attentions if there are in it

        if labels is not None:
            weight = [0.5,1-0.8,1-0.1/3,1-0.1/3,1-0.1/3]
            class_weight = torch.FloatTensor(weight).to(device)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0,weight=class_weight)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # return (loss), logits, (mems), (hidden states), (attentions)

def evaluation(MAX_LEN, tokenizer, test_loader,model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     
    prediction = pd.DataFrame(columns=['name_pred'])
    with torch.no_grad():
        ids, pred = [], []
        for data in test_loader: 
            input_ids = data.to(device) 
            outputs = model(input_ids)
            ans_pred = torch.sigmoid(outputs[0])
            _, ans_pred = ans_pred.max(-1)
            ans_pred = list(ans_pred[0])
            ans_index = [i for i in range(len(ans_pred)) if ans_pred[i] ==1 and ans_pred[i+1]==2]
            text = [tokenizer.convert_ids_to_tokens(i) for i in input_ids]
            text = list(text[0])
            ans_tokens = [text[x] for i,x in enumerate(ans_index)]
            
            ans_tokens=[]
            a = ''
            for i in range(len(ans_index)):
                index = ans_index[i]
                if ans_pred[index].item() == 1:
                    a = text[index]
                    for j in range(1,4):
                        if ans_pred[index+j].item() == 3 or index > MAX_LEN-3:
                            a += text[index+j]
                            break
                        else:
                            a += text[index+j]    
                ans_tokens.append(a)
            
            names, answer = [], []
            if ans_tokens != []:
                ws_results = ws([text])
                pos_results = pos(ws_results)
                ner_results = ner(ws_results, pos_results)
                for n in ner_results[0]:
                    if n[2] == 'PERSON':
                        names.append(n[3])
            names = list(set(names))
            for t in ans_tokens:
                for n in names:
                    if t in n and n not in answer and len(n) < 5 and len(n) > 1:
                        if len(n) == 2 and n[1] in ['嫌','男','女']:
                            print('discard',n)
                        elif len(n) == 4 and n[0]+n[1]+n[2] in names:
                            print('discard',n)
                        else:
                            answer.append(n)    
            pred.append(answer)
            
        prediction['name_pred'] = pred
    return prediction

def main(argv, arc):
    
    #test_path = argv[1]
    #output_path = argv[2]   
    PRETRAINED_MODEL_NAME = "hfl/chinese-xlnet-base"
    tokenizer = XLNetTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, do_lower_case=True)
    config = XLNetConfig.from_pretrained(PRETRAINED_MODEL_NAME)
    model = XLNetForTokenClassification(config)       
    model = model.cuda()
    checkpoint = torch.load('xlnet_6')
    model.load_state_dict(checkpoint)
    model = model.eval()
    MAX_LEN = 600
    BS=1
    test = TestingDataset(MAX_LEN, tokenizer, 'test.txt')
    testloader = torch.utils.data.DataLoader(test, batch_size=BS,shuffle=False)
    prediction = evaluation(MAX_LEN, tokenizer, testloader, model)
    output = prediction['name_pred'][0]
    print(output)
    
if __name__ == '__main__':
    main(sys.argv, len(sys.argv))