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

df_real=pd.read_csv(r"Dataset\zomato.csv")
# df_real.head()
# df_real['dish_liked'].isna()
# df_real.isnull().sum()
df=df_real.drop(['url','phone','menu_item'],axis=1)
# df.describe()
# df.info()
# df.isnull().count()#gives the count of nonnull values
# df.isnull().sum()#return the no of null values
df=df.drop(['dish_liked'],axis=1)
df.dropna(how='any',inplace=True)
# df.info()
# df.duplicated().sum()
df.drop_duplicates(inplace=True)
df = df.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type', 'listed_in(city)':'city'})
# df.head()
# df['votes'].isna().sum()
# df['cost']
df['cost']=df['cost'].astype(str)
df['cost']=df['cost'].apply(lambda x:x.replace(',','.'))
df['cost']=df['cost'].astype(float)
df['cost']
df_real['rate'].unique()

df=df.loc[df.rate!='NEW']
df=df.loc[df.rate!='-'].reset_index(drop=True)
def remove_slash(x):
    if isinstance(x, str):
        return x.replace('/5', '').strip()
    return x

df['rate'] =df['rate'].apply(remove_slash).astype(float)

# df['rate'].unique()
df.name=df.name.apply(lambda x:x.title())
df.online_order.replace(('Yes','No'),(True,False),inplace=True)
df.book_table.replace(('Yes','NO'),(True,False),inplace=True)
# df['online_order']
# df['book_table']
# df.isnull().sum()#return the no of null values
# df.describe()
# df.info()
# df['rate'].count()#Total no of rating values in the DataFrame
# df['rate'].unique()#Total no of unique rating value in the DataFrame
# df['rate'].unique().sum()
# df.head()
# df['type'].count()
# df['type'].unique()
# df['reviews_list']
# rs_name=list(df['name'].unique())
# rs_name
# df.columns
# df.columns
# df.city.unique()
def citybased(city):
    df['city']=df['city'].str.lower()
    citybase=df[df['city']==city.lower()]
    citybase=citybase.sort_values(by='rate',ascending=False)
    citybase.drop_duplicates(subset='address',keep='first',inplace=True)
    if(citybase.empty==0):
        hname=citybase[['address', 'name', 'cuisines','online_order', 'book_table', 'rate']]
        e = pd.DataFrame(hname.head(10))
        html_table = e.to_html(classes='table table-bordered', index=False)
        return (html_table)
    else:
        print('No Hotels Available')

