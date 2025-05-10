import numpy as np
import pandas as pd
import json
import os
import requests
from dotenv import load_dotenv
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import pickle

# Download NLTK data (runs once on startup)
nltk.download('punkt', quiet=True)

load_dotenv()
secret_key = str(os.getenv('secret_key'))

# Load datasets
movies = pd.read_csv("./tmdb_5000_movies.csv")
credits = pd.read_csv("./tmdb_5000_credits.csv")
movies = movies.merge(credits, on='title')

# Drop unnecessary columns
columns = ["budget", "homepage", "original_language", "original_title", "popularity", "production_companies", "release_date", "revenue", "runtime", "spoken_languages", "status", "tagline", "vote_average", "production_countries", "id", "vote_count"]
movies.drop(columns=columns, inplace=True)

# Clean data
movies.dropna(inplace=True)

# Convert JSON columns
def convert(obj):
    L = []
    obj = json.loads(obj)
    for i in range(len(obj)):
        L.append(obj[i]["name"])
    return L

movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)

def convert_3(obj):
    L = []
    obj = json.loads(obj)
    counter = 0
    for i in range(len(obj)):
        if counter < 3:
            L.append(obj[i]["name"])
            counter += 1
        else:
            break
    return L

movies["cast"] = movies["cast"].apply(convert_3)

def convert_director(obj):
    L = []
    obj = json.loads(obj)
    for i in range(len(obj)):
        if obj[i]["job"] == "Director":
            L.append(obj[i]["name"])
            return L
    return L

movies["crew"] = movies["crew"].apply(convert_director)

# Preprocess tags
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

movies["tags"] = movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]

# Create new DataFrame with explicit copy
new_df = movies[["movie_id", "title", "tags"]].copy()  # Use .copy() to avoid view issues

# Apply transformations using .loc
new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: x.lower())
ps = PorterStemmer()
new_df.loc[:, 'tags'] = new_df['tags'].apply(stem)

# Vectorize and compute similarity
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

# Save similarity matrix to avoid recomputation
with open('similarity.pkl', 'wb') as f:
    pickle.dump(similarity, f)

def fetchFromApi(movie_id):
    response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={secret_key}")
    data = response.json()
    return {
        "poster_path": "https://image.tmdb.org/t/p/w185/" + data["poster_path"],
        "title": data["title"]
    }

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    movies_data = []
    for i in movies_list:
        movies_data.append(fetchFromApi(int(new_df.iloc[i[0]].movie_id)))
    return movies_data