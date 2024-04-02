from m1_main import *

import pandas as pd
import numpy as np

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Get the topics and the top words for each topic
def get_topic_keywords(model, topn_words):
    topic_keywords = []
    topic_keywords_prob = []
    for topic_idx, topic in enumerate(model.components_):
        topic_keywords.append([feature_names[i] for i in topic.argsort()[:-topn_words - 1:-1]])
        topic_keywords_prob.append([topic[i] for i in topic.argsort()[:-topn_words - 1:-1]])

    return topic_keywords, topic_keywords_prob

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

df = pd.read_pickle(preprocessed_df_path)
df['CTB_word_tokens'] = df['CTB_word_tokens'].apply(lambda row: ' '.join(row))

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.95, min_df=0.05, max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['CTB_word_tokens'])
feature_names = tfidf_vectorizer.get_feature_names_out()

# Range of topics to try
num_topics_range = range(2, 10)  # Adjust the range as needed

# Lists to store reconstruction errors and silhouette scores
reconstruction_errors = []
silhouette_scores = []

for num_topics in num_topics_range:
    nmf_model = NMF(n_components=num_topics, random_state=100)
    nmf_model.fit(tfidf_matrix)
    
    # Reconstruction error
    reconstruction_error = nmf_model.reconstruction_err_
    reconstruction_errors.append(reconstruction_error)
    
    # Silhouette score
    topic_probabilities = nmf_model.transform(tfidf_matrix)
    silhouette = silhouette_score(topic_probabilities, np.argmax(topic_probabilities, axis=1))
    silhouette_scores.append(silhouette)

# Plotting the elbow curve
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(num_topics_range, reconstruction_errors, marker='o')
plt.xlabel('Number of Topics')
plt.ylabel('Reconstruction Error')
plt.title('Elbow Method for Optimal Number of Topics')

# Plotting the silhouette scores
plt.subplot(1, 2, 2)
plt.plot(num_topics_range, silhouette_scores, marker='o')
plt.xlabel('Number of Topics')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal Number of Topics')

plt.tight_layout()
plt.show()