import streamlit as st
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load data
df = pd.read_csv('./main_hr.csv')

# Load spaCy's English model with word vectors
try:
    nlp = spacy.load('en_core_web_md')
except OSError:
    from spacy.cli import download
    download('en_core_web_md')
    nlp = spacy.load('en_core_web_md')

# Function to calculate word vectors
def calculate_word_vectors(text):
    doc = nlp(text)
    # Average the word vectors for all tokens in the document
    return np.mean([token.vector for token in doc if not token.is_stop and not token.is_punct], axis=0)

# Function to process user input and return recommendations
def helper(user_prompt: str):
    user_prompt_doc = nlp(user_prompt)

    # Concatenate 'position' and 'tags' columns
    features = df[['name', 'contact', 'position', 'years_exp', 'tags']].copy()
    features['combined_text'] = features['name'] + ' ' + features['position'] + ' ' + features['tags']

    # Calculate word vectors for combined text
    features['word_vectors'] = features['combined_text'].apply(lambda x: calculate_word_vectors(str(x)))

    # Calculate cosine similarity between user prompt and each row
    features['cosine_similarity'] = features['word_vectors'].apply(
        lambda x: cosine_similarity([x], [user_prompt_doc.vector])[0][0])

    # Sort the DataFrame by cosine similarity scores
    features_sorted = features.sort_values(by='cosine_similarity', ascending=False)

    # Add a 'rank' column to indicate the rank of each recommendation
    features_sorted['rank'] = range(1, len(features_sorted) + 1)

    # Recommend top 5 columns
    top_5_recommendations = features_sorted.head(5)

    return top_5_recommendations[['name', 'contact', 'position', 'years_exp']].to_dict(orient='records')


# ==============================================================

# Streamlit app
def main():
    st.title("Recommendation System")

    # User input
    user_input = st.text_input("Enter your prompt:")
    
    if st.button("Get Recommendations"):
        if user_input:
            data_output = helper(user_input)
            st.json(data_output)
        else:
            st.error("Please enter a prompt.")

if __name__ == "__main__":
    main()
