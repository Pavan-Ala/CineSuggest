# CineSuggest: Movie Recommendation System

## Overview
CineSuggest is a personalized movie recommendation system built with Python and Streamlit. It uses collaborative filtering (SVD) and content-based filtering to provide tailored movie recommendations based on user preferences and viewing history.

## Features
- Personalized movie recommendations based on user ID
- Genre-based filtering for more targeted suggestions
- Cold-start handling for new users
- Clean, minimalist UI with Streamlit

## Project Structure
```
├── model/                  # Pre-trained models and data
│   ├── genre_matrix.pkl    # Genre features matrix
│   ├── movie_ids.pkl       # Movie ID mapping
│   ├── movie_titles.pkl    # Movie titles mapping
│   ├── ratings.csv         # Sample ratings data
│   ├── svd_model.pkl       # Trained SVD model
│   ├── user_ids.pkl        # User ID mapping
│   └── user_item_sparse.pkl # Sparse user-item matrix
├── links.csv              # Movie ID mappings (TMDB, IMDB)
├── movies.csv             # Movie metadata with genres
├── ratings.csv            # User ratings data
├── requirements.txt       # Project dependencies
├── streamlit_app.py       # Main Streamlit application
└── tags.csv               # Movie tags data
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Pavan-Ala/CineSuggest.git
cd cinesuggest
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. Open your browser and navigate to http://localhost:8501

3. Enter your User ID, select a preferred genre (optional), and adjust the number of recommendations

4. Click "Get Recommendations" to see your personalized movie suggestions

## Technical Details

The recommendation system combines:
- Collaborative filtering using Singular Value Decomposition (SVD)
- Content-based filtering using movie genres
- Popularity-based recommendations for cold-start scenarios

## License

This project is licensed under the MIT License - see the LICENSE file for details.