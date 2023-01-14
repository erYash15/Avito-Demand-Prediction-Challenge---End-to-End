import os
import pickle
import random
import re

import cv2
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import tensorflow as tf
from dask.diagnostics import ProgressBar
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import (LSTM, BatchNormalization, Dense, Dropout,
                                     Embedding, Flatten, Input, concatenate)
from tensorflow.keras.preprocessing.image import (ImageDataGenerator,
                                                  img_to_array, load_img)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm
from wordcloud import WordCloud

ProgressBar().register()

import warnings
from config import *
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Loading Files

max_features_desc = 300000
max_length_desc = 200

max_features_title = 150000
max_length_title = 8

embedding_vector_size = 300

region_length = 29
city_length = 1644
parent_category_name_length = 10
category_name_length = 48
param_1_length = 360
param_2_length = 233
param_3_length = 965
user_type_length = 4
image_top_1_C_length = 3039
region__city_length = 1702
param_1__param_2__param_3_length = 1920
parent_category_name__user_type_length = 28
category_name__user_type_length = 141
region__user_type_length = 85

num_features = 17

with open("./bin/embedding_matrix_title.pkl", "rb") as input_file:
    embedding_matrix_title = pickle.load(input_file)
with open("./bin/embedding_matrix_desc.pkl", "rb") as input_file:
    embedding_matrix_desc = pickle.load(input_file)


def FE_pipeline(df):

    ## Creating duplicate features
    df['image_top_1_C'] = df['image_top_1']
    df['image_top_1_N'] = df['image_top_1']

    print("Duplicated columns created.")
    
    ## Filling missing value
    print("Filling missing value.")
    # Categorical and Text Feature
    df['param_1'] = df['param_1'].fillna('null')
    df['param_2'] = df['param_2'].fillna('null')
    df['param_3'] = 'null'
    df['description'] = df['description'].fillna('null')
    df['image_top_1_C'] = df['image_top_1_C'].astype(str).fillna('null')
    
    print("Missing value filled for categorical and text feature.")
    
    # Numerical Feature
    price_average_df = pd.read_csv('./bin/price_category_name_mean.csv')
    image_top_1_average_df = pd.read_csv('./bin/image_top_1_N_category_name_mean.csv')
    
    def fill_missing_price(df, price_average_df):
        df = df.merge(price_average_df, on='category_name', how='left').reset_index(drop=True)
        df.loc[df[df['price_x'].isnull() == True].index, 'price_x'] = df[df['price_x'].isnull() == True]['price_y']
        df = df.drop(columns=['price_y'])
        df.rename(columns = {'price_x':'price'}, inplace = True)
        return df
    
    def fill_missing_image_top_1(df, image_top_1_average_df):
        df = df.merge(image_top_1_average_df, on='category_name', how='left').reset_index(drop=True)
        df.loc[df[df['image_top_1_N_x'].isnull() == True].index, 'image_top_1_N_x'] = df[df['image_top_1_N_x'].isnull() == True]['image_top_1_N_y']
        df = df.drop(columns=['image_top_1_N_y'])
        df.rename(columns = {'image_top_1_N_x':'image_top_1_N'}, inplace = True)
        return df

    df = fill_missing_price(df, price_average_df)
    df = fill_missing_image_top_1(df, image_top_1_average_df)

    print("Missing value filled for numerical feature.")
    
    ## Interactive Features 
    df['region__city'] = df['region'] + df['city']
    df['param_1__param_2__param_3'] = df['param_1'] + df['param_2'] + df['param_3']
    df['parent_category_name__user_type'] = df['parent_category_name'] + df['user_type']
    df['category_name__user_type'] = df['category_name'] + df['user_type']
    df['region__user_type'] = df['region'] + df['user_type']

    print("Interactive features created.")
    
    ## More Features
    # calling the text_featuring function which extract features from the title and description.
    df["title_words_length"] = df["title"].apply(lambda x: len(x.split()))
    df["description_words_length"] = df["description"].apply(lambda x: len(x.split()))

    df['symbol1_count'] = df['description'].str.count('↓')
    df['symbol2_count'] = df['description'].str.count('\*')
    df['symbol3_count'] = df['description'].str.count('✔')
    df['symbol4_count'] = df['description'].str.count('❀')
    df['symbol5_count'] = df['description'].str.count('➚')
    df['symbol6_count'] = df['description'].str.count('ஜ')
    df['symbol7_count'] = df['description'].str.count('.')
    df['symbol8_count'] = df['description'].str.count('!')
    df['symbol9_count'] = df['description'].str.count('\?')
    df['symbol10_count'] = df['description'].str.count('  ')
    df['symbol11_count'] = df['description'].str.count('-')
    df['symbol12_count'] = df['description'].str.count(',')
    
    print("More features created.")
    
    ## Feature Engineering
    # Categorical Features
    
    def categorical_encoder_P(Series, name): # Pipeline
        '''This function encode the categorical feature which we will use in NN along with embedding layer'''
        tokeniser = pickle.load(open('./bin/'+name+'_tokeniser.pkl', 'rb'))
        Series = np.array(tokeniser.texts_to_sequences(Series)).astype(np.int32)
        Series = Series[:,0]
        return Series
    
    df['region'] = categorical_encoder_P(df['region'], 'region') #1
    df['city'] = categorical_encoder_P(df['city'], 'city') #2
    df['parent_category_name'] = categorical_encoder_P(df['parent_category_name'], 'parent_category_name') #3
    df['category_name'] = categorical_encoder_P(df['category_name'], 'category_name') #4
    df['param_1'] = categorical_encoder_P(df['param_1'], 'param_1') #5
    df['param_2'] = categorical_encoder_P(df['param_2'], 'param_2') #6
    df['param_3'] = categorical_encoder_P(df['param_3'], 'param_3') #7
    df['user_type'] = categorical_encoder_P(df['user_type'], 'user_type') #8

    df['region__city'] = categorical_encoder_P(df['region__city'], 'region__city') #9
    df['param_1__param_2__param_3'] = categorical_encoder_P(df['param_1__param_2__param_3'], 'param_1__param_2__param_3') #10
    df['parent_category_name__user_type'] = categorical_encoder_P(df['parent_category_name__user_type'], 'parent_category_name__user_type') #11
    df['category_name__user_type'] = categorical_encoder_P(df['category_name__user_type'], 'category_name__user_type') #12
    df['region__user_type'] = categorical_encoder_P(df['region__user_type'], 'region__user_type') #13
    df['image_top_1_C'] = categorical_encoder_P(df['image_top_1_C'], 'image_top_1_C') #14
    
    print("Encoding Categorical Features.")
    
    # Numerical Feature
    
    columns = ['price', 'item_seq_number', 'image_top_1_N', 'title_words_length', 
             'description_words_length',
             'symbol1_count', 'symbol2_count', 'symbol3_count', 'symbol4_count',
             'symbol5_count', 'symbol6_count', 'symbol7_count', 'symbol8_count',
             'symbol9_count', 'symbol10_count', 'symbol11_count', 'symbol12_count']

    # In this subsection we have will transform the numerical features. 
    # Applying logirthmic transformation to avoid bais caused by normalization. Then batch_normalization layer.

    for col in columns:
        df[col] = df[col].apply(lambda x: np.log10(x+1)) # adding 1 as bias

    print("Numerical Features transformed")

    # Text Feature Engineering
    
    def text_clean(text):
        '''This function clean the russian text'''
        text = str(text)
        text = text.lower()
        clean = re.sub(r"[,.;@#?!&$-]+\ *", " ", text)
        return clean

    df['title'] = df['title'].apply(text_clean)
    df['description'] = df['description'].apply(text_clean)
    
    print("Text title and description are cleaned.")
    
    with open("./bin/tokenizer_title.pkl", "rb") as input_file:
        tokenizer_title = pickle.load(input_file)
    with open("./bin/tokenizer_desc.pkl", "rb") as input_file:
        tokenizer_desc = pickle.load(input_file)

    def encoder(train, tokenizer, max_length):
        ''' This function perform the tokenization and then convert words to integers and then perform padding and returns the values '''
        encoded_str = tokenizer.texts_to_sequences(train)
        padded_str = np.array(pad_sequences(encoded_str, maxlen=max_length, padding='post')).astype(np.float64)
        return padded_str

    padded_title_df = encoder(df['title'], tokenizer_title, 8)
    padded_desc_df = encoder(df['description'], tokenizer_desc, 200)

        
    return df, padded_title_df, padded_desc_df 


def Data_Preprocessing(df):
    
    df = df.drop(columns=['item_id', 'user_id', 'title', 'description'])

    df_cat = df[['region', 'city', 'parent_category_name', 'category_name',
                 'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1_C',
                 'region__city', 'param_1__param_2__param_3',
                 'parent_category_name__user_type', 'category_name__user_type',
                 'region__user_type']]
    
    df_num = df[['price', 'item_seq_number', 'image_top_1_N', 'title_words_length', 
                 'description_words_length',
                 'symbol1_count', 'symbol2_count', 'symbol3_count', 'symbol4_count',
                 'symbol5_count', 'symbol6_count', 'symbol7_count', 'symbol8_count',
                 'symbol9_count', 'symbol10_count', 'symbol11_count', 'symbol12_count']]

    df = pd.concat([df_cat, df_num],axis=1)
    
    return df.values

# Loading Model

def model():

    input1 = Input(shape= (max_length_title,), name='title')
    x1 = Embedding(
        max_features_title,
        embedding_vector_size,
        weights = [embedding_matrix_title],
        trainable=False)(input1)
    x1 = LSTM(64, return_sequences=False)(x1)
    x1 = Flatten()(x1)

    input2 = Input(shape= (max_length_desc,), name='description')
    x2 = Embedding(
        max_features_desc,
        embedding_vector_size,
        weights = [embedding_matrix_desc],
        trainable=False)(input2)
    x2 = LSTM(64, return_sequences=False)(x2)
    x2 = Flatten()(x2)

    input3 = Input(shape= (num_features, ), name='numerical')
    x3 = BatchNormalization()(input3)
    x3 = Dense(64, activation='relu')(x3)
    x3 = Dense(16, activation='relu')(x3)


    inputc1 = Input(shape= (1,), name='region')
    c1 = Embedding(input_dim=region_length, output_dim=2, trainable=True)(inputc1)
    c1 = Flatten()(c1)

    inputc2 = Input(shape= (1,), name='city')
    c2 = Embedding(input_dim=city_length, output_dim=2, trainable=True)(inputc2)
    c2 = Flatten()(c2)

    inputc3 = Input(shape= (1,), name='parent_category_name')
    c3 = Embedding(input_dim=parent_category_name_length, output_dim=2, trainable=True)(inputc3)
    c3 = Flatten()(c3)

    inputc4 = Input(shape= (1,), name='category_name')
    c4 = Embedding(input_dim=category_name_length, output_dim=2, trainable=True)(inputc4)
    c4 = Flatten()(c4)

    inputc5 = Input(shape= (1,), name='param_1')
    c5 = Embedding(input_dim=param_1_length, output_dim=2, trainable=True)(inputc5)
    c5 = Flatten()(c5)

    inputc6 = Input(shape= (1,), name='param_2')
    c6 = Embedding(input_dim=param_2_length, output_dim=2, trainable=True)(inputc6)
    c6 = Flatten()(c6)

    inputc7 = Input(shape= (1,), name='param_3')
    c7 = Embedding(input_dim=param_3_length, output_dim=2, trainable=True)(inputc7)
    c7 = Flatten()(c7)

    inputc8 = Input(shape= (1,), name='user_type')
    c8 = Embedding(input_dim=user_type_length, output_dim=2, trainable=True)(inputc8)
    c8 = Flatten()(c8)

    inputc9 = Input(shape= (1,), name='image_top_1_C')
    c9 = Embedding(input_dim=image_top_1_C_length, output_dim=2, trainable=True)(inputc9)
    c9 = Flatten()(c9)

    inputc10 = Input(shape= (1,), name='region__city')
    c10 = Embedding(input_dim=region__city_length, output_dim=2, trainable=True)(inputc10)
    c10 = Flatten()(c10)

    inputc11 = Input(shape= (1,), name='param_1__param_2__param_3')
    c11 = Embedding(input_dim=param_1__param_2__param_3_length, output_dim=2, trainable=True)(inputc11)
    c11 = Flatten()(c11)

    inputc12 = Input(shape= (1,), name='parent_category_name__user_type')
    c12 = Embedding(input_dim=parent_category_name__user_type_length, output_dim=2, trainable=True)(inputc12)
    c12 = Flatten()(c12)

    inputc13 = Input(shape= (1,), name='category_name__user_type')
    c13 = Embedding(input_dim=category_name__user_type_length, output_dim=2, trainable=True)(inputc13)
    c13 = Flatten()(c13)

    inputc14 = Input(shape= (1,), name='region__user_type')
    c14 = Embedding(input_dim=region__user_type_length, output_dim=2, trainable=True)(inputc14)
    c14 = Flatten()(c14)

    output = concatenate([x1, x2, x3, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14])
    output = Dropout(0.3)(output)
    output = Dense(128, activation='relu')(output)
    output = Dropout(0.3)(output)
    output = Dense(64, activation='relu')(output)
    output = Dropout(0.3)(output)
    output = Dense(32, activation='relu')(output)
    output = Dense(1, activation='softmax')(output)

    model = Model(inputs = [input1, input2, input3, inputc1, inputc2, inputc3, inputc4, inputc5, inputc6, inputc7, inputc8, inputc9, inputc10, inputc11, inputc12, inputc13, inputc14], outputs = output)

    model.compile(loss='mse', 
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01,clipnorm=1.0,clipvalue=0.05), 
        metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    try:
        model.load_weights('./models/best_model_4.h5')
        print("Model Loaded!")
    except:
        print("No Model Available")

    return model


model = model()


def final_fun_1(X):

    print("Raw_Data:", X.shape)
    print("="*50)
    
    X_test, padded_title_test, padded_desc_test  = FE_pipeline(X)
    X_test = Data_Preprocessing(X_test)

    print("="*50)
    print("Summary of all Fetaures")
    print("="*50)
    print("1. X_test:", X_test.shape)
    print("2. test Title Encoded", padded_title_test.shape)
    print("3. test Description Encoded", padded_desc_test.shape)
    
    region_test = X_test[:,0]
    city_test = X_test[:,1]
    parent_category_name_test = X_test[:,2]
    category_name_test = X_test[:,3]
    param_1_test = X_test[:,4]
    param_2_test = X_test[:,5]
    param_3_test = X_test[:,6]
    user_type_test = X_test[:,7]
    image_top_1_C_test = X_test[:,8]
    region__city_test = X_test[:,9]
    param_1__param_2__param_3_test = X_test[:,10]
    parent_category_name__user_type_test = X_test[:,11]
    category_name__user_type_test = X_test[:,12]
    region__user_type_test = X_test[:,13]

    ntest = X_test[:,14:]

    # predicting on batch caused memory error prediction value by value
    
    try:
        ypred = model.predict([padded_title_test, padded_desc_test, ntest, 
                    region_test, city_test, parent_category_name_test, 
                    category_name_test, param_1_test, param_2_test, param_3_test, 
                    user_type_test, image_top_1_C_test, region__city_test, 
                    param_1__param_2__param_3_test, parent_category_name__user_type_test, 
                    category_name__user_type_test, region__user_type_test])
    except:
        print("Failed!")
    
    return y_pred


@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())

@app.route('/predict', methods=['POST', "GET"])
def predict():
    
    item_id = str(request.form['item_id'])
    user_id = str(request.form['user_id'])
    region = str(request.form['region'])
    city = str(request.form['city'])
    parent_category_name = str(request.form['parent_category_name'])
    category_name = str(request.form['category_name'])

    param_1 = str(request.form['param_1'])
    param_2 = str(request.form['param_2'])
    param_3 = str(request.form['param_3'])

    if param_3 == 'nan':
        param_3 = np.nan

    title = str(request.form['title'])

    description = str(request.form['description'])

    price = float(request.form['price'])
    item_seq_number = int(request.form['item_seq_number'])
    user_type = str(request.form['user_type'])
    image_top_1 = float(request.form['image_top_1'])

    result = 'hi'
    df = pd.DataFrame({'item_id':[item_id],
                        'user_id':[user_id],
                        'region':[region],
                        'city':[city],
                        'parent_category_name':[parent_category_name],
                        'category_name':[category_name],
                        'param_1':[param_1],
                        'param_2':[param_2],
                        'title':[title],
                        'description':[description],
                        'price':[price],
                        'item_seq_number':[item_seq_number],
                        'user_type':[user_type],
                        'image_top_1':[image_top_1]
                        })

    ypred = final_fun_1(df)

    result = y_pred

    return render_template('index.html', **locals())


if __name__ == '__main__':
    app.run(debug=True)