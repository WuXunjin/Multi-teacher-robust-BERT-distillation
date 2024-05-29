# importing libraries 
import datetime
import pandas as pd 
import numpy as np
import tensorflow as tf
import os 
os.chdir('/home/zjlab/Multi/')
from Utils.utils import data_tokenization
from transformers import AutoTokenizer
from Utils.data_loader import data_loader
from Bert.bert_model import bert_model
from sklearn.metrics import classification_report,accuracy_score
from MeanTeacher.train_mean_teacher import train_mean_teacher
strategy = tf.distribute.MirroredStrategy()
import random 
random.seed(42)

data_name='IMDB'  
Data_loc='/home/zjlab/Multi/Data/IMDB/'

#Adjusting feature as per data 
if data_name =='codalab':
    feature_col='tweet'
    target_col='label'
    aug_col='aug_tweet'
    class_1='real'
    class_2='fake'
elif data_name =='IMDB':
    feature_col='text'
    target_col='label'
    aug_col='aug_text'
    class_1=1
    class_2=0

max_len = 150
learning_rate = 2e-5
epochs= 3
batch_size= 4

df_train,df_test,df_aug_unlabel= data_loader(Data_loc ,target_col ,aug_col ,class_1 )
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
x_train,y_train,vocab_size= data_tokenization(df_train,feature_col ,target_col ,max_len ,tokenizer)
x_test,y_test,_= data_tokenization(df_test,feature_col ,target_col ,max_len ,tokenizer)
x_unlabel,_,_=data_tokenization(df_aug_unlabel,aug_col,target_col,max_len,tokenizer)
# print('df_train',df_train)
# print('df_test',df_test)
# print('df_aug_unlabel',df_aug_unlabel)
# print('x_train',x_train)
# print('y_train',y_train)
# print('x_test',x_test)
# print('y_test',y_test)
# print('x_unlabel',x_unlabel)
student= train_mean_teacher(x_train,y_train,x_unlabel,epochs,batch_size,learning_rate,max_len)
student.save_weights('SavedModels/IMDB_cls+dis+con.h5')
# load weights
#student = bert_model('distilbert-base-uncased',max_len,learning_rate)
#student.load_weights("SavedModels/IMDB_cls+dis+con.h5")

logits_nat = student.predict(x_test)
print('Natural accuracy', accuracy_score(np.argmax(y_test,1),np.argmax(logits_nat,1),digits=4))


#PGD Adversarial Training
epsilon = 0.1
alpha = 0.01
num_iter = 10
def pgd_attack(x, y, epsilon, alpha, num_iter):
    x_adv = tf.identity(x)
    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            logits = student.predict(x_adv)
            loss = tf.keras.losses.categorical_crossentropy(y, logits)
        gradients = tape.gradient(loss, x_adv)
        gradients = tf.sign(gradients)
        x_adv = tf.clip_by_value(x_adv + alpha * gradients, x - epsilon, x + epsilon)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
    return x_adv

x_adv = pgd_attack(x_test, y_test, epsilon, alpha, num_iter)
logits_adv = student.predict(x_adv)
print('Adversarial accuracy', accuracy_score(np.argmax(y_test,1),np.argmax(logits_adv,1),digits=4))
