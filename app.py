
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sys import stderr
import pandas as pd

df = pd.read_csv('./main_hr.csv')

try:
    nlp = spacy.load('en_core_web_md')
except OSError:
    print('Downloading language model for the spaCy POS tagger\n'
          "(don't worry, this will only happen once)", file=stderr)
    from spacy.cli import download
    download('en_core_web_md')
    nlp = spacy.load('en_core_web_md')

# Load spaCy's English model with word vectors
nlp = spacy.load("en_core_web_md")


def calculate_word_vectors(text):
    doc = nlp(text)
    # Average the word vectors for all tokens in the document
    return np.mean([token.vector for token in doc if not token.is_stop and not token.is_punct], axis=0)


def helper(user_prompt: str):
    user_prompt_doc = nlp(user_prompt)

    # Concatenate 'position' and 'tags' columns
    features = df[['name','contact', 'position', 'years_exp','tags']].copy()
    features['combined_text'] = features['name'] + ' ' + \
        features['position'] + ' ' + features['tags']

    # Calculate word vectors for combined text
    features['word_vectors'] = features['combined_text'].apply(
        lambda x: calculate_word_vectors(str(x)))

    # Calculate cosine similarity between user prompt and each row
    features['cosine_similarity'] = features['word_vectors'].apply(
        lambda x: cosine_similarity([x], [user_prompt_doc.vector])[0][0])

    # Sort the DataFrame by cosine similarity scores
    features_sorted = features.sort_values(
        by='cosine_similarity', ascending=False)

    # Add a 'rank' column to indicate the rank of each recommendation
    features_sorted['rank'] = range(1, len(features_sorted) + 1)

    # Recommend top 5 columns
    top_5_recommendations = features_sorted.head(5)

    # Display top 5 recommendations with ranks
    # print(top_5_recommendations[['rank', 'name','contact', 'position', 'years_exp']])

    top_5_recommendations.reset_index(drop=True, inplace=True)

    send_data = []

    for i in range(0, 5):
        send_data.append(
            {
                "name": f"{top_5_recommendations['name'][i]}",
                "position": f"{top_5_recommendations['position'][i]}",
                "contact": f"{top_5_recommendations['contact'][i]}",
                "yoe": f"{top_5_recommendations['years_exp'][i]}"
            }
        )
    return send_data



# ============================================================
from flask import Flask, render_template, request
from flask_cors import CORS
import random
import json

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['get', 'post'])
def main():
    if request.method == 'POST':
        userInput = request.json['data']
        dataOutput = helper(userInput)
        return {
            "output": dataOutput
        }
    return render_template('index.html')

# if(__name__ == "__main__"):
#     app.run(debug=True, port=8000)