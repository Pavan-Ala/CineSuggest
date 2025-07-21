import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
import requests

# Check for naming conflicts
if os.path.exists('pandas.py') or os.path.exists('numpy.py') or os.path.exists('scipy.py') or os.path.exists('sklearn.py') or os.path.exists('tqdm.py'):
    st.error("Error: Found file(s) with names conflicting with library imports (e.g., pandas.py). Please rename or remove them.")
    sys.exit(1)

# Load pre-trained model and data
try:
    svd_model = joblib.load('model/svd_model.pkl')
    user_item_sparse = joblib.load('model/user_item_sparse.pkl')
    user_ids = joblib.load('model/user_ids.pkl')
    movie_ids = joblib.load('model/movie_ids.pkl')
    genre_matrix = joblib.load('model/genre_matrix.pkl')
    movie_titles = joblib.load('model/movie_titles.pkl')
    ratings = pd.read_csv('model/ratings.csv')
    movies = pd.read_csv('movies.csv')
    st.success("Model and data loaded successfully.")
except FileNotFoundError as e:
    st.error(f"Error: {e}. Ensure 'model/' directory contains all saved files.")
    sys.exit(1)

# Extract unique genres
unique_genres = set()
for genres in movies['genres']:
    unique_genres.update(genres.split('|'))
unique_genres = sorted(list(unique_genres))

# Load links for TMDB IDs
links = pd.read_csv('links.csv')

def get_recommendations(user_id, svd_model, user_item_sparse, user_ids, movie_ids, movie_titles, genre_matrix, ratings, links, movies, n_recommendations=5, content_weight=0.1, preferred_genre=None, api_key=None):
    try:
        user_idx = list(user_ids).index(user_id)
        # SVD-based scores
        user_factors = svd_model.transform(user_item_sparse[user_idx])
        svd_scores = np.dot(user_factors, svd_model.components_).flatten()

        # Content-based scores
        user_ratings = ratings[ratings['userId'] == user_id]
        rated_movie_ids = user_ratings['movieId'].values
        rated_movie_indices = [list(movie_ids).index(mid) for mid in rated_movie_ids if mid in movie_ids]
        if not rated_movie_indices:
            content_scores = np.zeros(len(movie_ids))
        else:
            user_genre_profile = genre_matrix[rated_movie_indices].mean(axis=0)
            content_scores = cosine_similarity([user_genre_profile], genre_matrix)[0]

        # Combine scores
        combined_scores = (1 - content_weight) * svd_scores + content_weight * content_scores

        # Exclude rated items
        rated_items = user_item_sparse[user_idx].nonzero()[1]
        combined_scores[rated_items] = -np.inf

        # Get top N indices
        top_indices = np.argsort(combined_scores)[::-1]

        # Filter by preferred genre if provided
        candidate_movies = pd.DataFrame({'movieId': movie_ids, 'score': combined_scores})
        candidate_movies = candidate_movies.merge(movies[['movieId', 'genres']], on='movieId')
        if preferred_genre and preferred_genre != 'No preference':
            candidate_movies = candidate_movies[candidate_movies['genres'].str.contains(preferred_genre)]
        top_movie_ids = candidate_movies.sort_values('score', ascending=False)['movieId'].head(n_recommendations).tolist()
    except ValueError:
        # Cold start: popularity-based, filtered by genre
        popularity = ratings.groupby('movieId')['rating'].mean().reset_index()
        popularity = popularity.merge(movies[['movieId', 'genres']], on='movieId')
        if preferred_genre and preferred_genre != 'No preference':
            popularity = popularity[popularity['genres'].str.contains(preferred_genre)]
        top_movie_ids = popularity.sort_values('rating', ascending=False)['movieId'].head(n_recommendations).tolist()

    recommendations = pd.DataFrame({
        'movieId': top_movie_ids,
        'title': [movie_titles.get(mid, "Unknown") for mid in top_movie_ids]
    })
    return recommendations, None

# Streamlit UI
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&display=swap');
        .main {{
            background: linear-gradient(to right, #e0eafc, #cfdef3);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        h1 {{
            font-family: 'Playfair Display', serif;
            color: #2c3e50;
            text-align: center;
        }}
        .stButton>button {{
            background-color: #2980b9;
            color: white;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            border-radius: 6px;
            transition: background-color 0.3s;
        }}
        .stButton>button:hover {{
            background-color: #2c3e50;
        }}
        .stTextInput>div>div>input, .stNumberInput>div>div>input {{
            border: 1px solid #bdc3c7;
            border-radius: 6px;
            padding: 8px;
            font-size: 14px;
        }}
        .stSelectbox>div>div>select {{
            border: 1px solid #bdc3c7;
            border-radius: 6px;
            padding: 8px;
            font-size: 14px;
        }}
        .stSlider>div>div>div {{
            background-color: #2980b9;
        }}
        .recommendation-item {{
            background-color: white;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 10px;
            margin: 10px 0;
            font-size: 16px;
        }}
    </style>
""", unsafe_allow_html=True)

st.title("CineSuggest")
st.markdown("Enter your details to get personalized movie recommendations.")

user_id = st.number_input("User ID", min_value=1, step=1, help="Enter your unique user identifier")
preferred_genre = st.selectbox("Preferred Genre", ['No preference'] + unique_genres, help="Select your favorite genre for better recommendations")
n_recommendations = st.slider("Number of Recommendations", 1, 20, 5, help="Choose how many movies to recommend")

if st.button("Get Recommendations"):
    recommendations, error = get_recommendations(user_id, svd_model, user_item_sparse, user_ids, movie_ids, movie_titles, genre_matrix, ratings, links, movies, n_recommendations, preferred_genre=preferred_genre)
    if error:
        st.error(error)
    else:
        st.subheader(f"Recommendations for User {user_id}:")
        for _, row in recommendations.iterrows():
            st.markdown(f'<div class="recommendation-item">{row["title"]}</div>', unsafe_allow_html=True)
    st.dataframe(recommendations.style.format({"score": "{:.4f}"}))