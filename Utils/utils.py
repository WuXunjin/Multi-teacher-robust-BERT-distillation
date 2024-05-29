
from transformers import AutoTokenizer
import pandas as pd 
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow as tf



def dataset_split(dataset,test_size):
  df_train=dataset.sample(frac=(1-test_size),random_state=200) #random state is a seed value
  df_test=dataset.drop(df_train.index)
  df_train= df_train.reset_index(drop=True)
  df_test=df_test.reset_index(drop=True)
  return df_train,df_test

def data_tokenization(dataset,feature_col,target_col,max_len,tokenizer):
    tokens = dataset[feature_col].apply(lambda x: tokenizer(x,return_tensors='tf', 
                                                            truncation=True,
                                                            padding='max_length',
                                                            max_length=max_len, 
                                                            add_special_tokens=True))
    input_ids= []
    attention_mask=[]
    for item in tokens:
        input_ids.append(item['input_ids'])
        attention_mask.append(item['attention_mask'])
    input_ids, attention_mask=np.squeeze(input_ids), np.squeeze(attention_mask)

    # if we have label column
    if (target_col in dataset.columns):
        y= to_categorical(dataset[target_col],2)
        return [input_ids,attention_mask], y, tokenizer.vocab_size
    else:
        return [input_ids,attention_mask]


    