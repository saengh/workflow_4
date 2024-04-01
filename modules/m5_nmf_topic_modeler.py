from m1_main import *

import pandas as pd
import numpy as np

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

# Get the topics and the top words for each topic
def get_topic_keywords(model, topn_words):
  
  topic_keywords = []
  topic_keywords_prob = []
  for topic_idx, topic in enumerate(model.components_):
      topic_keywords.append([feature_names[i] for i in topic.argsort()[:-topn_words - 1:-1]])
      topic_keywords_prob.append([topic[i] for i in topic.argsort()[:-topn_words - 1:-1]])

    #   topic = model.components_[topic_idx]
    # top_word_indices = topic.argsort()[:-n_top_words - 1:-1]
    # top_words_with_probs = [(feature_names[i], topic[i]) for i in top_word_indices]
    # sorted_top_words = sorted(top_words_with_probs, key=lambda x: x[1], reverse=True)
    # return [word[0] for word in sorted_top_words]

  return topic_keywords, topic_keywords_prob

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------

df = pd.read_pickle(preprocessed_df_path)
df['CTB_word_tokens'] = df['CTB_word_tokens'].apply(lambda row: ' '.join(row))

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.95, min_df=0.05, max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['CTB_word_tokens'])
feature_names = tfidf_vectorizer.get_feature_names_out()

nmf_model = NMF(n_components=25, random_state=100)
nmf_model.fit(tfidf_matrix)

topn_words = 10
topic_keywords, topic_keywords_prob = get_topic_keywords(nmf_model, topn_words)

topic_probabilities = nmf_model.transform(tfidf_matrix)
dominant_topic_num = [topic.argmax() for topic in topic_probabilities]
dominant_topic_prob = [f'{topic.max(): .2%}' for topic in topic_probabilities]
dominant_topic_keywords_prob = [topic_keywords_prob[idx] for idx in dominant_topic_num]
dominant_topic_keywords = [topic_keywords[idx] for idx in dominant_topic_num]

# # Print the topics and top words
# for i, topic_words in enumerate(topic_keywords):
#     print(f"Topic {i + 1}:")
#     print(", ".join(topic_words))

df['Topic_num'] = dominant_topic_num
df['Topic_prob'] = dominant_topic_prob
df['Topic_keywords_prob'] = dominant_topic_keywords_prob
df['Topic_keywords'] = dominant_topic_keywords

columns_to_drop = [f"{field}_{token_type}" for field in fields for token_type in token_type_list]
df = df.drop(columns=columns_to_drop, errors='ignore')

df.to_excel(workflow_folder + r'\excel\nmf_topics.xlsx', index=False)
df.to_pickle(workflow_folder + r'\pickle\nmf_topics.pickle')