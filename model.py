import sys
import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import joblib

# Check for naming conflicts
if os.path.exists('pandas.py') or os.path.exists('numpy.py') or os.path.exists('scipy.py') or os.path.exists('sklearn.py') or os.path.exists('tqdm.py'):
    print("Error: Found file(s) with names conflicting with library imports (e.g., pandas.py). Please rename or remove them.")
    sys.exit(1)

# Reuse functions from sample1.py
def load_data(dataset_folder=r"C:\\Users\\pavan\\OneDrive\\Documents\\recommender system\\ml-32m"):
    ratings_path = os.path.join(dataset_folder, "ratings.csv")
    movies_path = os.path.join(dataset_folder, "movies.csv")
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Dataset folder not found at: {dataset_folder}")
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"Ratings file not found at: {ratings_path}")
    if not os.path.exists(movies_path):
        raise FileNotFoundError(f"Movies file not found at: {movies_path}")
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    movie_titles = dict(zip(movies['movieId'], movies['title']))
    print(f"Loaded ratings: {ratings.shape[0]} rows")
    print(f"Loaded movies: {movies.shape[0]} rows")
    return ratings, movies, movie_titles

def clean_data(ratings):
    ratings = ratings.drop(columns=['timestamp'])
    if ratings[['userId', 'movieId', 'rating']].isnull().any().any():
        print("Warning: Missing values found. Dropping rows with missing values.")
        ratings = ratings.dropna(subset=['userId', 'movieId', 'rating'])
    if (ratings['rating'] < 0.5).any() or (ratings['rating'] > 5.0).any():
        print("Warning: Invalid ratings found. Filtering to valid range [0.5, 5.0].")
        ratings = ratings[(ratings['rating'] >= 0.5) & (ratings['rating'] <= 5.0)]
    if ratings.duplicated(['userId', 'movieId']).any():
        print("Warning: Duplicate user-movie ratings found. Keeping first occurrence.")
        ratings = ratings.drop_duplicates(['userId', 'movieId'], keep='first')
    print(f"Cleaned ratings: {ratings.shape[0]} rows")
    return ratings

def filter_data(ratings, min_user_ratings=200, min_movie_ratings=300):
    user_counts = ratings['userId'].value_counts()
    active_users = user_counts[user_counts >= min_user_ratings].index
    ratings = ratings[ratings['userId'].isin(active_users)]
    movie_counts = ratings['movieId'].value_counts()
    popular_movies = movie_counts[movie_counts >= min_movie_ratings].index
    ratings = ratings[ratings['movieId'].isin(popular_movies)]
    print(f"Filtered ratings shape: {ratings.shape}")
    print(f"Number of unique users: {ratings['userId'].nunique()}")
    print(f"Number of unique movies: {ratings['movieId'].nunique()}")
    return ratings

def create_user_item_matrix(ratings):
    try:
        user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
        user_means = user_item_matrix.mean(axis=1)
        user_item_matrix = user_item_matrix.sub(user_means, axis=0).fillna(0)
        user_item_sparse = csr_matrix(user_item_matrix.values)
        print(f"User-item matrix shape: {user_item_matrix.shape}")
        return user_item_matrix, user_item_sparse, user_item_matrix.index, user_item_matrix.columns
    except MemoryError:
        print("MemoryError: Using sparse matrix only.")
        ratings['userId'] = ratings['userId'].astype('category')
        ratings['movieId'] = ratings['movieId'].astype('category')
        user_ids = ratings['userId'].cat.categories
        movie_ids = ratings['movieId'].cat.categories
        user_means = ratings.groupby('userId')['rating'].mean()
        ratings['rating'] = ratings.apply(lambda x: x['rating'] - user_means[x['userId']], axis=1)
        user_item_sparse = csr_matrix((ratings['rating'], (ratings['userId'].cat.codes, ratings['movieId'].cat.codes)))
        return None, user_item_sparse, user_ids, movie_ids

def create_genre_matrix(movies, movie_ids):
    genres = set()
    for genre_str in movies['genres']:
        genres.update(genre_str.split('|'))
    genres = sorted(list(genres))
    if '(no genres listed)' in genres:
        genres.remove('(no genres listed)')
    genre_matrix = np.zeros((len(movie_ids), len(genres)))
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}
    for _, row in movies.iterrows():
        if row['movieId'] in movie_id_to_idx:
            movie_idx = movie_id_to_idx[row['movieId']]
            movie_genres = row['genres'].split('|')
            for genre in movie_genres:
                if genre in genres:
                    genre_matrix[movie_idx, genres.index(genre)] = 1
    transformer = TfidfTransformer()
    genre_matrix = transformer.fit_transform(genre_matrix).toarray()
    print(f"Genre matrix shape: {genre_matrix.shape}")
    return genre_matrix, genres

def main():
    try:
        # Load and preprocess data
        dataset_folder = r"C:\\Users\\pavan\\OneDrive\\Documents\\recommender system\\ml-32m"
        ratings, movies, movie_titles = load_data(dataset_folder)
        ratings = clean_data(ratings)
        ratings = filter_data(ratings, min_user_ratings=200, min_movie_ratings=300)
        user_item_matrix, user_item_sparse, user_ids, movie_ids = create_user_item_matrix(ratings)
        genre_matrix, genres = create_genre_matrix(movies, movie_ids)

        # Train SVD model
        n_components = 100  # Use best n_components from tuning
        svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        svd_model.fit(user_item_sparse)

        # Save model and data
        os.makedirs('model', exist_ok=True)
        joblib.dump(svd_model, 'model/svd_model.pkl')
        joblib.dump(user_item_sparse, 'model/user_item_sparse.pkl')
        joblib.dump(user_ids, 'model/user_ids.pkl')
        joblib.dump(movie_ids, 'model/movie_ids.pkl')
        joblib.dump(genre_matrix, 'model/genre_matrix.pkl')
        joblib.dump(movie_titles, 'model/movie_titles.pkl')
        ratings.to_csv('model/ratings.csv', index=False)
        print("Model and data saved successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Ensure dataset folder and files are correct and dependencies are installed.")

if __name__ == "__main__":
    main()