import transformers
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import os
import pandas as pd 
import numpy as np 
from transformers import MarianMTModel, MarianTokenizer
from tensorflow.keras.models import Model, Sequential
import random
import nltk
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
random.seed(42)

main_dir='Data/'

# dataset names are codalab, IMDB,...
dataset_folder_name='IMDB'

feature_col='text'
target_col='label' 
aug_col='aug_text'
perturb_col= 'Perturbation%'

# augmentation parameter 
max_word_aug=10 # adjust this parameter to change the word level augmentation
max_len=100

pretrained_weights= 'distilbert-base-uncased'

proc_train_loc='/home/zjlab/Multi/Data/IMDB/PreprocessedData/pr_train.csv'
test_loc='/home/zjlab/Multi/Data/IMDB/PreprocessedData/pr_test.csv'

#splitting the dataset 
def df_splitter(df,num_of_partition=3):
    partition_ratio= 1/num_of_partition
    collection=[]

    for i in range(num_of_partition):
        random_state= np.random.choice(np.random
                                       .randint(1,100))
        collection.append(df.sample(frac=partition_ratio,
                                    replace=True,
                                    random_state=random_state)
                                    .reset_index(drop=True))
    return collection

# Synonyn Augmentation
def synonym_augment(df,max_aug=10,aug_src='wordnet', iter=2):
    print('Executing synonym augmentation')
    df_aug=pd.DataFrame()
    aug=naw.SynonymAug(aug_src=aug_src,
                       aug_max=max_aug)
    for i in range(iter):
        df[aug_col]= aug.augment(list(df[feature_col].values))
        df_aug=pd.concat([df_aug, df], ignore_index=True)

    return df_aug.reset_index(drop=True)

#Context Augmentation
def context_augment(df,pretrained_model='distilbert-base-uncased',action='substitute',iter=1):

    print('Executing context augmentation')
    aug=naw.ContextualWordEmbsAug(model_path=pretrained_model,action=action)
    df_aug=pd.DataFrame()
    for i in range(iter):
        df[aug_col]= aug.augment(list(df[feature_col].values))
        df_aug=pd.concat([df_aug, df], ignore_index=True)
    
    return df_aug.reset_index(drop=True)


def back_translation(df):
    print('Executing Backtranslation augmentation')

    # english to romance
    target_model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'
    tokenizer = MarianTokenizer.from_pretrained(target_model_name)
    model = MarianMTModel.from_pretrained(target_model_name)
    
    #romance to english
    en_model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'
    en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
    en_model = MarianMTModel.from_pretrained(en_model_name)
    
    def translated_texts(text):
        translated = model.generate(**tokenizer([text], return_tensors="pt", padding=True))
        rom_texts = tokenizer.decode(translated[0], skip_special_tokens=True) 

        back_translated = en_model.generate(**en_tokenizer(rom_texts, return_tensors="pt", padding=True))
        translated_texts = en_tokenizer.decode(back_translated[0], skip_special_tokens=True).lower()

        return translated_texts

    df_back_translation= df[(df[feature_col].str.len()>0)].reset_index(drop=True)
    df_back_translation[aug_col]=df_back_translation[feature_col].apply(lambda row: translated_texts(row))

    return df_back_translation


def data_cleaning(df):
    
    print(f'Initial Length of the dataframe: {len(df)}')
    print(f'Number of same augmented text  and original text:{len(df)-len(df[~(df[feature_col]==df[aug_col])])}')
    df=df[~(df[feature_col]==df[aug_col])].reset_index(drop=True)
    print(f'After removinal : {len(df)}')

    # unique words replacement
    df['set_aug']= df[aug_col].apply(lambda row: len(set(row.split(' '))))
    df=df[~(df['set_aug']<3)].reset_index(drop=True)
    print(f'Length after dropping less than 3 unique words in the augtweet: {len(df)}')
    

    return df[df.columns.difference(['set_aug'])]

adv_unlabel_df= pd.read_csv(proc_train_loc,header='infer',usecols=[feature_col,target_col]) 
adv_unlabel_df= adv_unlabel_df.dropna().reset_index(drop=True)
df_syn,df_context,df_back_translated=df_splitter(adv_unlabel_df)

# Creating Unlabel data 

for i in range(2,20):
    df_b_temp= df_syn[i*100:(i+1)*100]
    df_b_temp= synonym_augment(df_b_temp)
    df_b_temp=data_cleaning(df_b_temp)
    if os.path.exists('Data/IMDB/AugmentedData/adv_test_syn.csv'):
        df_temp_1= pd.read_csv('Data/IMDB/AugmentedData/adv_test_syn.csv')
        df_b_temp = pd.concat([df_b_temp, df_temp_1])
        df_b_temp.to_csv('Data/IMDB/AugmentedData/adv_test_syn.csv', index=False)
    else:
        df_b_temp.to_csv('Data/IMDB/AugmentedData/adv_test_syn.csv', index=False)
for i in range(2,20):
    df_b_temp= df_context[i*100:(i+1)*100]
    df_b_temp= context_augment(df_b_temp)
    df_b_temp=data_cleaning(df_b_temp)
    if os.path.exists('Data/IMDB/AugmentedData/adv_test_context.csv'):
        df_temp_1= pd.read_csv('Data/IMDB/AugmentedData/adv_test_context.csv')
        df_b_temp = pd.concat([df_b_temp, df_temp_1])
        df_b_temp.to_csv('Data/IMDB/AugmentedData/adv_test_context.csv', index=False)
    else:
        df_b_temp.to_csv('Data/IMDB/AugmentedData/adv_test_context.csv', index=False)
for i in range(2,20):
    df_b_temp= df_back_translated[i*100:(i+1)*100]
    df_b_temp= back_translation(df_b_temp)
    df_b_temp=data_cleaning(df_b_temp)
    if os.path.exists('Data/IMDB/AugmentedData/adv_test_backTrans.csv'):
        df_temp_1= pd.read_csv('Data/IMDB/AugmentedData/adv_test_backTrans.csv')
        df_b_temp = pd.concat([df_b_temp, df_temp_1])
        df_b_temp.to_csv('Data/IMDB/AugmentedData/adv_test_backTrans.csv', index=False)
    else:
        df_b_temp.to_csv('Data/IMDB/AugmentedData/adv_test_backTrans.csv', index=False)

def perturb_cal(row):
  list1=row[feature_col].split()
  list2=row[aug_col].split()
  count=0
  for w1,w2 in zip(list1,list2):
    if w1!=w2:
      count+=1
  perturb_per= round((count/len(list1))*100,3)
  return perturb_per

adv_test_syn_df= pd.read_csv('Data/IMDB/AugmentedData/adv_test_syn.csv')
adv_test_context_df= pd.read_csv('Data/IMDB/AugmentedData/adv_test_context.csv')
adv_test_backTrans_df= pd.read_csv('Data/IMDB/AugmentedData/adv_test_backTrans.csv')
adv_test_syn_df[perturb_col]=adv_test_syn_df.apply(lambda row: perturb_cal(row),axis=1)
print('advtest',adv_test_syn_df)

# from sklearn.feature_extraction.text import TfidfVectorizer
# import gensim.downloader as api
# from semantic_text_similarity.models import WebBertSimilarity
# from semantic_text_similarity.models import ClinicalBertSimilarity

# wmd_model = api.load('word2vec-google-news-300')
# vectorizer = TfidfVectorizer()
# web_model = WebBertSimilarity(device='cpu', batch_size=10)

# def wmd(text1,text2):
#   return  wmd_model.wmdistance(text1, text2)

# def cosine_sim(text1, text2):
#   tfidf = vectorizer.fit_transform([text1, text2])
#   return ((tfidf * tfidf.T).A)[0,1]

# def semantic_sim(text1,text2):
#   return web_model.predict([(text1,text2)])[0]
  
# def similarity_exploration(df):

#   df['CosineSim']= df.apply(lambda row: cosine_sim(row[aug_col],row[feature_col]),axis=1)
#   df['Wmd']= df.apply(lambda row: wmd(row[aug_col],row[feature_col]),axis=1)
#   df['Semantic_sim']= df.apply(lambda row: semantic_sim(row[aug_col],row[feature_col]),axis=1)
#   return df

# adv_test_syn_df= similarity_exploration(adv_test_syn_df)
# adv_test_context_df=similarity_exploration(adv_test_context_df)
# adv_test_backTrans_df=similarity_exploration(adv_test_backTrans_df)  

# # saving the file
# adv_test_syn_df.to_csv('IMDB/AugmentedData/adv_test_syn.csv', index=False)
# adv_test_context_df.to_csv('IMDB/AugmentedData/adv_test_context.csv', index=False)
# adv_test_backTrans_df.to_csv('IMDB/AugmentedData/adv_test_backTrans.csv', index=False)


