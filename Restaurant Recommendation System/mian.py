import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
# from keras.layers.core import Dense,Activation,Dropout
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from keras.callbacks import EarlyStopping
import math
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
hotel_details=pd.read_csv(r'DataSet\Hotel_details.csv')
hotel_rooms=pd.read_csv(r'DataSet\Hotel_Room_attributes.csv')
hotel_cost=pd.read_csv(r'DataSet\hotels_RoomPrice.csv')

hotel_details.head()
hotel_rooms.head()
del hotel_details['id']
del hotel_rooms['id']
del hotel_details['zipcode']
hotel_details=hotel_details.dropna()
hotel_rooms=hotel_rooms.dropna()
hotel_details.drop_duplicates(subset='hotelid',keep=False,inplace=True)
hotel=pd.merge(hotel_rooms,hotel_details,left_on='hotelcode',right_on='hotelid',how='inner')
hotel['hotelcode'].isna().count()
hotel.columns
hotel['roomtype']
def citybased(city):
    hotel['city']=hotel['city'].str.lower()
    citybase=hotel[hotel['city']==city.lower()]
    citybase=citybase.sort_values(by='starrating',ascending=False)
    citybase.drop_duplicates(subset='hotelcode',keep='first',inplace=True)
    if(citybase.empty==0):
        hname=citybase[['hotelname','starrating','address','roomamenities','ratedescription']]
        df = pd.DataFrame(hname.head(10))

        # Convert DataFrame to HTML
        html_table = df.to_html(classes='table table-bordered', index=False)
        return (html_table)

    else:
        print('No Hotels Available')


# citybased('Borovets')

# print(hname.head)
