import pandas as pd 
import tensorflow as tf 
import numpy as np 


def data_loader(Data_loc,target_col, aug_col, class_1):
  df_train_loc=Data_loc+'PreprocessedData/pr_train.csv'
  df_test_loc= Data_loc+'PreprocessedData/pr_test.csv' 
  df_train= pd.read_csv(df_train_loc,index_col=[0])
  df_train= df_train.dropna()
  df_train= df_train.drop_duplicates().reset_index(drop=True)
  i = 0
  while i < len(df_train[target_col]):
    if(df_train[target_col][i]==class_1):
      df_train.loc[i,target_col]=1
    else:
      df_train.loc[i,target_col]=0
    i=i+1
  # Test data 
  df_test= pd.read_csv(df_test_loc,index_col=[0],encoding='ISO-8859-1')
  df_test= df_test.dropna()
  df_test= df_test.drop_duplicates().reset_index(drop=True)
  i = 0
  while i < len(df_test[target_col]):
    if(df_test[target_col][i]==class_1):
      df_test.loc[i,target_col]=1
    else:
      df_test.loc[i,target_col]=0
    i=i+1
  # loading unlabel data 
  # Adversarial unlabeled data location 
  df_aug_syn_un_loc=Data_loc+'AugmentedData/aug_synonym.csv'
  df_aug_con_un_loc=Data_loc+'AugmentedData/aug_context.csv'
  df_aug_bt_un_loc=Data_loc+'AugmentedData/aug_backtranslated.csv'
  # Reading adversarial Unlabel data 
  df_aug_syn_un= pd.read_csv(df_aug_syn_un_loc,encoding='ISO-8859-1')
  df_aug_syn_un= df_aug_syn_un.dropna().drop_duplicates().reset_index(drop=True)
  df_aug_con_un= pd.read_csv(df_aug_con_un_loc,encoding='ISO-8859-1')
  df_aug_con_un= df_aug_con_un.dropna().drop_duplicates().reset_index(drop=True)
  df_aug_bt_un= pd.read_csv(df_aug_bt_un_loc,encoding='ISO-8859-1')
  df_aug_bt_un= df_aug_bt_un.dropna().drop_duplicates().reset_index(drop=True)
  # Combining all together 
  df_aug_unlabel = df_aug_syn_un._append(df_aug_con_un)._append(df_aug_bt_un)
  df_aug_unlabel= df_aug_unlabel.sample(frac=1).reset_index(drop=True)
  i = 0
  while i < len(df_aug_unlabel[target_col]):
    if(df_aug_unlabel[target_col][i]==class_1):
      df_aug_unlabel.loc[i,target_col]=1
    else:
      df_aug_unlabel.loc[i,target_col]=0
    i=i+1
  return df_train, df_test, df_aug_unlabel
