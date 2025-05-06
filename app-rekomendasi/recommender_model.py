import os
import numpy as np
import pandas as pd

class TFContentBasedRecommender:
    def __init__(self, model_path='../models/tf_content_recommender'):
        self.model_path = model_path
        self.embedding_dim = 512
        self.articles_df = None
        self.article_embeddings = None

   

    def load_model(self):
        try:
            self.article_embeddings = np.load(os.path.join(self.model_path, 'article_embeddings.npy'))
            self.articles_df = pd.read_pickle(os.path.join(self.model_path, 'articles_df.pkl'))
            return True
        except:
            return False

    def recommend(self, article_id, top_n=5):
        if self.articles_df is None or self.article_embeddings is None:
            raise ValueError("Model not loaded")

        idx = self.articles_df[self.articles_df['UUID'] == article_id].index[0]
        query_embedding = self.article_embeddings[idx].reshape(1, -1)

        similarity = np.dot(self.article_embeddings, query_embedding.T).flatten()
        norms = np.linalg.norm(self.article_embeddings, axis=1) * np.linalg.norm(query_embedding)
        similarity = similarity / norms

        sim_indices = np.argsort(similarity)[::-1][1:top_n+1]
        recommended_articles = self.articles_df.iloc[sim_indices][['UUID', 'title', 'province', 'city']].copy()
        recommended_articles['similarity_score'] = similarity[sim_indices]

        return recommended_articles.to_dict(orient='records')
