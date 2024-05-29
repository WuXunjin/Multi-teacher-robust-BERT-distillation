from Bert.bert_model import bert_model
max_len = 150
learning_rate = 2e-5

model = bert_model('distilbert-base-uncased',max_len,learning_rate)
model.load_weights('/home/zjlab/Multi/SavedModels/cls+dis+con.h5')

import textattack
from textattack.models.wrappers import ModelWrapper
import transformers
import pandas as pd 
import numpy as np
import torch
import tensorflow as tf

data_name = 'codalab'
feature_col='tweet'
target_col='label'
pretrained_weights = 'distilbert-base-uncased'

df_test_for_attack= pd.read_csv('Data/codalab/df_adv_test.csv')
#df_test_for_attack= pd.read_csv('Data/IMDB/PreprocessedData/pr_test.csv')
dataset_for_attack=list(df_test_for_attack.itertuples(index=False, name=None))
dataset_for_attack = textattack.datasets.Dataset(dataset_for_attack)

class CustomTensorFlowModelWrapper(ModelWrapper):
    def __init__(self, model,pretrained_weights):
        self.model = model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_weights)
    def __call__(self, text_input_list):
        tokens=self.tokenizer(text_input_list,return_tensors='tf', 
                         truncation=True,
                         padding='max_length',
                         max_length=max_len, 
                         add_special_tokens=True)
        input_ids= []
        attention_mask=[]
        input_ids.append(tokens.input_ids)
        attention_mask.append(tokens.attention_mask)
        preds = torch.tensor(self.model([input_ids, attention_mask]).numpy())
  
        return preds

model_wrapper = CustomTensorFlowModelWrapper(model,pretrained_weights)
tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')
attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
#attack = textattack.attack_recipes.TextBuggerLi2018.build(model_wrapper)
#attack = textattack.attack_recipes.BAEGarg2019.build(model_wrapper)
attack_args = textattack.AttackArgs(
    num_examples=100,
    log_to_csv="log.csv",
    checkpoint_interval=5,
    disable_stdout=False,
    query_budget=200
)
attacker = textattack.Attacker(attack, dataset_for_attack, attack_args)
attacker.attack_dataset()
