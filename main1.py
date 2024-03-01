import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import re
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder

zomato_real=pd.read_csv("Dataset/zomato.csv")
# zomato_real.head() # prints the first 5 rows of the dataset
zomato=zomato_real.drop(['url','phone'],axis=1) #Dropping the column "dish_liked", "phone", "url" and saving the new dataset as "zomato"
zomato.duplicated().sum()
zomato.drop_duplicates(inplace=True)
zomato.dropna(how='any',inplace=True)
zomato = zomato.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type', 'listed_in(city)':'city'})
#Some Transformations
zomato['cost'] = zomato['cost'].astype(str) #Changing the cost to string

zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',','.')) #Using lambda function to replace ',' from cost
zomato['cost'] = zomato['cost'].astype(float)
zomato = zomato.loc[zomato.rate !='NEW']
zomato = zomato.loc[zomato.rate !='-'].reset_index(drop=True)
def remove_slash(x):
    if isinstance(x, str):
        return x.replace('/5', '')
    else:
        return x

zomato['rate'] = zomato['rate'].apply(remove_slash).str.strip().astype(float)

# Adjust the column names
zomato.name = zomato.name.apply(lambda x:x.title())
zomato.online_order.replace(('Yes','No'),(True, False),inplace=True)
zomato.book_table.replace(('Yes','No'),(True, False),inplace=True)

## Computing Mean Rating
restaurants = list(zomato['name'].unique())
zomato['Mean Rating'] = 0

for i in range(len(restaurants)):
    zomato['Mean Rating'][zomato['name'] == restaurants[i]] = zomato['rate'][zomato['name'] == restaurants[i]].mean()
    
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (1,5))
zomato[['Mean Rating']] = scaler.fit_transform(zomato[['Mean Rating']]).round(2)

## Lower Casing
zomato["reviews_list"] = zomato["reviews_list"].str.lower()

## Removal of Puctuations
import string
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_punctuation(text))

## Removal of Stopwords

import nltk
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_stopwords(text))



## Removal of URLS
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_urls(text))

# RESTAURANT NAMES:
restaurant_names = list(zomato['name'].unique())
def get_top_words(column, top_nu_of_words, nu_of_word):
    vec = CountVectorizer(ngram_range= nu_of_word, stop_words='english')
    bag_of_words = vec.fit_transform(column)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:top_nu_of_words]


# Randomly sample 60% of your dataframe
df_percent = zomato.sample(frac=0.5)


df_percent.set_index('name', inplace=True)
indices = pd.Series(df_percent.index)

# Creating tf-idf matrix
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.0, stop_words='english')

tfidf_matrix = tfidf.fit_transform(df_percent['reviews_list'])
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend(name, cosine_similarities=cosine_similarities):
    try:
        idx_candidates = indices[indices == name].index

        if not idx_candidates.empty:
            idx = idx_candidates[0]
        else:
            print(f'Restaurant "{name}" not found.')
            return None

        score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
        top30_indexes = list(score_series.iloc[0:31].index)

        recommend_restaurant = [list(df_percent.index)[each] for each in top30_indexes]

        # Creating the new data set to show similar restaurants
        df_new_list = []

        for each in recommend_restaurant:
            df_temp = df_percent.loc[df_percent.index == each, ['address', 'cuisines', 'Mean Rating', 'online_order', 'cost']].sample()

            # Add a new column 'name' with the value of the original index
            df_temp['name'] = each
            
            df_new_list.append(df_temp)

        # Concatenate the list of DataFrames into a single DataFrame
        df_new = pd.concat(df_new_list, ignore_index=True)

        # Reorder columns to have 'name' as the first column
        df_new = df_new[['name', 'address', 'cuisines', 'Mean Rating', 'online_order', 'cost']]

        # Drop duplicates and sort by Mean Rating
        df_new = df_new.drop_duplicates(subset=['name', 'address', 'cuisines', 'Mean Rating', 'online_order', 'cost'])
        df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(1000)

        # print(f'TOP {len(df_new)} RESTAURANTS LIKE {name} WITH SIMILAR REVIEWS: ')
        e=pd.DataFrame( df_new.head(10))
        html_table = e.to_html(classes='table table-bordered', index=False)
        return (html_table)
        
    except Exception as e:
        print(f'An error occurred: {str(e)}')
        return None

# Assuming df_percent is your DataFrame and indices is the indices Series
# Make sure to replace these with your actual DataFrame and indices

# print( recommend('Spice Elephant'))

# Assuming your dataset is stored in a DataFrame named 'zomato'
# Make sure to replace 'your_dataset.csv' with the actual file path if reading from a CSV file
# zomato = pd.read_csv('your_dataset.csv')

# Extract relevant columns
data = zomato[['name', 'rate']]

# Drop rows with missing values in the 'rate' column
data = data.dropna(subset=['rate'])

# Encode restaurant names using LabelEncoder
le = LabelEncoder()
data['name_encoded'] = le.fit_transform(data['name']).astype(str)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['name_encoded']], data['rate'], test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer for restaurant names
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train['name_encoded'])
X_test_tfidf = vectorizer.transform(X_test['name_encoded'])

# Train a linear regression model
model = LinearRegression()
model.fit(X_train_tfidf, y_train)

def predict_rating_for_restaurant(restaurant_name):
    # Encode the input restaurant name
    restaurant_name_encoded = le.transform([restaurant_name]).astype(str)

    # Transform using the TF-IDF vectorizer
    restaurant_name_tfidf = vectorizer.transform(restaurant_name_encoded)

    # Predict the rating
    rating_prediction = model.predict(restaurant_name_tfidf)

    return rating_prediction[0]




# Example usage
restaurant_name_input = "San Churro Cafe"
predicted_rating = predict_rating_for_restaurant(restaurant_name_input)
print(f"Predicted Rating for {restaurant_name_input}: {predicted_rating}")

