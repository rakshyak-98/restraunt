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

df=df_real.drop(['url','phone','menu_item'],axis=1)

df=df.drop(['dish_liked'],axis=1)
df.dropna(how='any',inplace=True)

df.drop_duplicates(inplace=True)
df = df.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type', 'listed_in(city)':'city'})

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

def locationbased(city):
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

def cuisinesbased(cuisines):
    df['cuisines']=df['cuisines'].str.lower()
    cuisinesbase=df[df['cuisines']==cuisines.lower()]
    cuisinesbase=cuisinesbase.sort_values(by='rate',ascending=False)
    cuisinesbase.drop_duplicates(subset='address',keep='first',inplace=True)
    if(cuisinesbase.empty==0):
        hname=cuisinesbase[['address', 'name', 'cuisines','online_order', 'book_table', 'rate']]
        e=pd.DataFrame( hname.head(10))
        html_table = e.to_html(classes='table table-bordered', index=False)
        return (html_table)
    else:
        print('No Hotels Available')

# print(cuisinesbased("cafe"))
        
data=pd.read_csv(r"C:\Users\ASUSU\PycharmProjects\Zomato\zomato_restaurants_in_India.csv")
selected_columns = ['name', 'address', 'city', 'cuisines', 'aggregate_rating', 'delivery', 'takeaway']
new_df = data[selected_columns]
new_df['delivery'] = new_df['delivery'].replace({-1: 'No', 1: 'Yes'})
new_df['takeaway'] = new_df['takeaway'].replace({-1: 'No', 1: 'Yes'})
new_df.drop_duplicates(inplace=True)
new_df.dropna(how='any',inplace=True)
new_df = new_df.rename(columns={'name':'Name','address':'Address', 'city':'City','cuisines':'Cuisines','aggregate_rating':'Rating','delivery':'Delivery','takeaway':'Takeaway'})
def citybased(city):
    new_df['City']=new_df['City'].str.lower()
    citybase=new_df[new_df['City']==city.lower()]
    citybase=citybase.sort_values(by='Rating',ascending=False)
    citybase.drop_duplicates(subset='Address',keep='first',inplace=True)
    if(citybase.empty==0):
        hname=citybase[[ 'Name', 'Address','City', 'Cuisines', 'Rating', 'Delivery',
       'Takeaway']]
        e = pd.DataFrame(hname.head(10))
        html_table = e.to_html(classes='table table-bordered', index=False)
        return (html_table)
    else:
        print('No Hotels Available')  
   

# print(citybased('agra'))