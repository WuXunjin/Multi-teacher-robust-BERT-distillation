#1. Digit removal 2. Lemmatization 3. URLS removal 4. Punctuation removal
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import texthero as hero
import re
from texthero import stopwords
import os 
from wordcloud import WordCloud
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 
from textblob import TextBlob,Word
from nltk.corpus import words
nltk.download('words')
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
# stopwords=stopwords.words('english')
from pathlib import Path
from spellchecker import SpellChecker

main_dir='Data/'

raw_data_loc=os.path.join(main_dir,'IMDB/RawData')
proc_data_loc= os.path.join(main_dir,'IMDB/PreprocessedData')

# feature declaration 
target_col='label'
feature_col='text'

#reading data
train_file_n='Train.csv'
test_file_n='Test.csv'
val_file_n='Valid.csv'

# processed file name
pr_train_name='pr_train.csv'
pr_test_name='pr_test.csv'
pr_val_name='pr_val.csv'

#reading dataframe
df_train= pd.read_csv(os.path.join(raw_data_loc,train_file_n), encoding='ISO-8859-1',header='infer',index_col=[0]).dropna().reset_index(drop=True)
df_test= pd.read_csv(os.path.join(raw_data_loc,test_file_n), encoding='ISO-8859-1',header='infer',index_col=[0]).dropna().reset_index(drop=True)
df_val= pd.read_csv(os.path.join(raw_data_loc,val_file_n), encoding='ISO-8859-1',header='infer',index_col=[0]).dropna().reset_index(drop=True)

print('reading dataframe',df_train.shape,df_test.shape,df_val.shape)


def remove_punctuation(words):
    spell = SpellChecker()
    spell.word_frequency.load_words(['covid', 'corona'])
    spell.known(['covid', 'corona'])
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        new_word = re.sub(r'_|-|ー', ' ', new_word)
        new_word= re.sub(r'^RT[\s]+', '', new_word)
        new_word= re.sub(r'https?:\/\/.*[\r\n]*', '', new_word)
        new_word=  re.sub(r'#', '', new_word)
        new_word = re.sub(r'[0-9]', '', new_word)
        if spell.unknown(new_word):
          if new_word != '':
              new_words.append(new_word)
        else:
          if new_word != '':
              new_words.append(spell.correction(new_word))

    return " ".join(new_words)

def lemma_per_pos(sent):
    tweet_tokenizer=TweetTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_list = [lemmatizer.lemmatize(w) for w in  tweet_tokenizer.tokenize(sent)]
    return remove_punctuation(lemmatized_list)

def df_preprocessing(df,feature_col):
    stop = set(stopwords.words('english'))
    df[feature_col]= (df[feature_col].pipe(hero.lowercase).pipe(hero.remove_urls).pipe(hero.remove_digits) )
    df[feature_col]= [lemma_per_pos(sent) for sent in df[feature_col]]
    return df


df_train_processed= df_preprocessing(df_train,feature_col)
df_train_processed= df_train_processed.dropna().drop_duplicates().reset_index(drop=True)

df_test_processed= df_preprocessing(df_test,feature_col)
df_test_processed= df_test_processed.dropna().drop_duplicates().reset_index(drop=True)

df_val_processed= df_preprocessing(df_val,feature_col)
df_val_processed= df_val_processed.dropna().drop_duplicates().reset_index(drop=True)

print('data preprocessing',df_train.shape,df_test.shape,df_val.shape)

df_train_processed['len']= df_train_processed[feature_col].str.split().map(lambda x: len(x))
df_train_processed.sort_values('len', ascending=False).reset_index(drop=True)
print('Max length: {}, Min length: {}, Average Length :{}'.format(max(df_train_processed['len']),min(df_train_processed['len']),int(df_train_processed['len'].mean())))
print('Count of tweet length less than 100 words:',df_train_processed[df_train_processed.len<=150].count())
df_train_processed=df_train_processed[(df_train_processed.len>6) & (df_train_processed.len<=100)].reset_index(drop=True)
print('Max length: {}, Min length: {}, Mean Length :{}'.format(max(df_train_processed['len']),min(df_train_processed['len']),int(df_train_processed['len'].mean())))
 
df_test_processed['len']= df_test_processed[feature_col].str.split().map(lambda x: len(x))
df_test_processed.sort_values('len', ascending=False).reset_index(drop=True)
print('Count of tweet length less than 100 words',df_test_processed[df_test_processed.len<=100].count())
df_test_processed=df_test_processed[(df_test_processed.len>6) & (df_test_processed.len<=100)].reset_index(drop=True)
print('Max length: {}, Min length: {}, Average Length :{}'.format(max(df_test_processed['len']),min(df_test_processed['len']),int(df_test_processed['len'].mean())))

df_val_processed['len']= df_val_processed[feature_col].str.split().map(lambda x: len(x))
df_val_processed.sort_values('len', ascending=False).reset_index(drop=True)
df_val_processed=df_val_processed[(df_val_processed.len>6) & (df_val_processed.len<=100)].reset_index(drop=True)
print('Max length: {}, Min length: {}, Average Length :{}'.format(max(df_val_processed['len']),min(df_val_processed['len']),int(df_val_processed['len'].mean())))

# Saving into files
df_train_processed.to_csv(os.path.join(proc_data_loc,pr_train_name),columns=[feature_col, target_col])
df_test_processed.to_csv(os.path.join(proc_data_loc,pr_test_name),columns=[feature_col,target_col])
df_val_processed.to_csv(os.path.join(proc_data_loc,pr_val_name),columns=[feature_col, target_col])

