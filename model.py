#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv("./tmdb_5000_movies.csv")
credits = pd.read_csv("./tmdb_5000_credits.csv")


# In[3]:


movies.shape


# In[4]:


credits.shape


# In[5]:


movies = movies.merge(credits,on = 'title')


# In[6]:


columns = ["budget","homepage","original_language","original_title","popularity","production_companies","release_date","revenue","runtime","spoken_languages","status","tagline","vote_average","production_countries","id","vote_count"]


# In[7]:


movies.drop(columns = columns ,inplace = True)


# In[8]:


movies.head(1)


# In[9]:


movies.isnull().sum()


# In[10]:


movies.dropna(inplace = True)
movies.shape


# In[11]:


movies.duplicated().sum()


# In[12]:


import json
def convert(obj):
    L = []
    obj = json.loads(obj)
    for i in range(len(obj)):
        L.append(obj[i]["name"])
    return L


# In[13]:


movies["genres"] = movies["genres"].apply(convert)
movies.head()


# In[14]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[15]:


movies.head()


# In[16]:


def convert_3(obj):
    L = []
    obj = json.loads(obj)
    counter = 0
    for i in range(len(obj)):
        if counter < 3:
            L.append(obj[i]["name"])
            counter+=1
        else:
            break
    return L


# In[17]:


movies["cast"] = movies["cast"].apply(convert_3)


# In[18]:


movies.head()


# In[19]:


def convert_director(obj):
    L = []
    obj = json.loads(obj)
    for i in range(len(obj)):
        if obj[i]["job"] == "Director":
            L.append(obj[i]["name"])
            return L
    return L


# In[20]:


movies["crew"] = movies["crew"].apply(convert_director)


# In[21]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[22]:


movies.head()


# In[23]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[24]:


movies.shape


# In[25]:


movies["tags"] = movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]


# In[26]:


movies.head(1)


# In[27]:


new_df = movies[["movie_id","title","tags"]]


# In[28]:


new_df


# In[29]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[30]:


new_df['tags'][0]


# In[31]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[32]:


import nltk
from nltk.stem.porter import PorterStemmer 
ps = PorterStemmer()


# In[33]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[34]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[35]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000,stop_words = 'english')
vectors = cv.fit_transform(new_df['tags']).toarray()


# In[36]:


cv.get_feature_names_out() 


# In[37]:


vectors


# In[38]:


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors) 


# In[39]:


sorted(list(enumerate(similarity[0])),reverse = True,key = lambda x:x[1])[1:6]


# In[48]:
import requests
import json
from dotenv import load_dotenv
load_dotenv()
import os
secret_key = str(os.getenv('secret_key'))
def fetchFromApi(movie_id):
    response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={secret_key}")
    data = response.json()
    return {
        "poster_path":"https://image.tmdb.org/t/p/w185/"+data["poster_path"],
        "title":data["title"]
    }

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse = True,key = lambda x:x[1])[1:6]
    movies_data = []
    for i in movies_list:
        movies_data.append(fetchFromApi(int(new_df.iloc[i[0]].movie_id)))
    return movies_data
