import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf

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

class TFCollaborativeRecommender:
    def __init__(self, model_path='../models/tf_collaborative_recommender'):
        self.model_path = model_path
        self.model = None
        self.user_mapping = {}
        self.article_mapping = {}
        self.inverse_article_mapping = {}
    def _create_model(self, num_users, num_articles, embedding_size=32):
        user_input = tf.keras.layers.Input(shape=(1,), name='user_input')
        article_input = tf.keras.layers.Input(shape=(1,), name='article_input')

        user_embedding = tf.keras.layers.Embedding(num_users, embedding_size, name='user_embedding')(user_input)
        article_embedding = tf.keras.layers.Embedding(num_articles, embedding_size, name='article_embedding')(article_input)

        user_bias = tf.keras.layers.Embedding(num_users, 1, name='user_bias')(user_input)
        article_bias = tf.keras.layers.Embedding(num_articles, 1, name='article_bias')(article_input)

        user_vecs = tf.keras.layers.Flatten()(user_embedding)
        article_vecs = tf.keras.layers.Flatten()(article_embedding)
        user_bias = tf.keras.layers.Flatten()(user_bias)
        article_bias = tf.keras.layers.Flatten()(article_bias)

        dot_product = tf.keras.layers.Dot(axes=1)([user_vecs, article_vecs])

        global_bias = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer='zeros', name='global_bias')(
            tf.keras.layers.Lambda(lambda x: x * 0 + 1)(user_input))

        output = tf.keras.layers.Add()([dot_product, user_bias, article_bias, global_bias])

        model = tf.keras.Model(inputs=[user_input, article_input], outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
        return model
    def load_model(self, embedding_size=32):
        if not os.path.exists(self.model_path):
            return False

        with open(os.path.join(self.model_path, 'user_mapping.json'), 'r') as f:
            self.user_mapping = {str(k): int(v) for k, v in json.load(f).items()}
        with open(os.path.join(self.model_path, 'article_mapping.json'), 'r') as f:
            self.article_mapping = {str(k): int(v) for k, v in json.load(f).items()}
        with open(os.path.join(self.model_path, 'global_bias.json'), 'r') as f:
            self.global_bias = float(json.load(f))

        self.reverse_user_mapping = {v: k for k, v in self.user_mapping.items()}
        self.reverse_article_mapping = {v: k for k, v in self.article_mapping.items()}

        self.user_factors = np.load(os.path.join(self.model_path, 'user_factors.npy'))
        self.article_factors = np.load(os.path.join(self.model_path, 'article_factors.npy'))
        self.user_biases = np.load(os.path.join(self.model_path, 'user_biases.npy'))
        self.article_biases = np.load(os.path.join(self.model_path, 'article_biases.npy'))

        num_users = len(self.user_mapping)
        num_articles = len(self.article_mapping)
        self.model = self._create_model(num_users, num_articles, embedding_size)
        self.model.load_weights(os.path.join(self.model_path, '.weights.h5'))

        return True
    
    def recommend_for_user(self, user_id: str, top_n: int = 5):
        
        if "u1e2f3g4-2222" not in self.user_mapping:
            raise ValueError("User ID tidak ditemukan dalam mapping.")

        user_idx = self.user_mapping[user_id]
        all_article_indices = list(self.article_mapping.values())
        user_array = np.array([user_idx] * len(all_article_indices))
        article_array = np.array(all_article_indices)
        user_array = user_array.reshape(-1, 1)
        article_array = article_array.reshape(-1, 1)

        scores = self.model.predict([user_array, article_array], verbose=0)

        scores = scores.flatten() 
        
        top_indices = np.argsort(scores)[::-1][:top_n]  # Mengurutkan skor tertinggi
        top_article_ids = [self.reverse_article_mapping[article_array[i][0]] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]

        return list(zip(top_article_ids, top_scores))
