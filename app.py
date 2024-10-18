import streamlit as st
import joblib
import pickle
# Load saved models and data
svd_model = joblib.load('svd_model.pkl')   # Load your SVD model
cosine_sim = joblib.load('cosine_sim.pkl') # Load the cosine similarity matrix
movies = pd.read_csv('movies.csv')


# Define the hybrid recommendation logic
def hybrid_recommendations(user_id, title, svd_model, cos_sim, movies, top_n=10):
    # Fuzzy matching or exact title search for the input movie
    if title not in movies['title'].values:
        st.write(f"Movie '{title}' not found in the dataset.")
        return []

    movie_idx = movies[movies['title'] == title].index[0]  # Get the index of the movie
    
    # Content-based filtering part
    sim_scores = list(enumerate(cos_sim[movie_idx]))  # Get similarity scores for the movie
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # Sort by similarity
    sim_scores = sim_scores[1:top_n+1]  # Exclude the first match (itself) and get top_n
    
    # Collaborative filtering part (predicted ratings using SVD)
    recommendations = []
    for i, score in sim_scores:
        movie_id = movies['movieId'].iloc[i]
        est_rating = svd_model.predict(user_id, movie_id).est  # Predict the rating for the movie
        recommendations.append((movies['title'].iloc[i], est_rating))
    
    # Sort recommendations by the predicted rating
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    
    return recommendations

# Streamlit UI
st.title("Hybrid Movie Recommendation System")

# Input for user_id and movie title
user_id = st.number_input("Enter User ID:", min_value=1, step=1)
movie_title = st.text_input("Enter Movie Title (e.g., Toy Story (1995)): ")

# Button to trigger recommendations
if st.button("Get Recommendations"):
    recommended_movies = hybrid_recommendations(user_id, movie_title, svd_model, cosine_sim, movies, top_n=5)
    
    # Display the recommended movies
    if recommended_movies:
        st.write("Recommended Movies:")
        for movie, rating in recommended_movies:
            st.write(f"{movie}")
    else:
        st.write("No recommendations found.")
