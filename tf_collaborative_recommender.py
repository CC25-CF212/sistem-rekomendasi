import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

class TFCollaborativeRecommender:
    def __init__(self):
        self.model = None
        self.user_encoder = LabelEncoder()
        self.article_encoder = LabelEncoder()
        self.user_mapping = {}
        self.article_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_article_mapping = {}
        self.user_factors = None
        self.article_factors = None
        self.global_bias = None
        self.user_biases = None
        self.article_biases = None
        self.model_path = 'models/tf_collaborative_recommender'
        
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

    def fit(self, interactions_df, embedding_size=32, epochs=20, batch_size=64, validation_split=0.1):
        self.user_encoder.fit(interactions_df['user_id'].unique())
        self.article_encoder.fit(interactions_df['article_id'].unique())

        self.user_mapping = dict(zip(interactions_df['user_id'].unique(), self.user_encoder.transform(interactions_df['user_id'].unique())))
        self.article_mapping = dict(zip(interactions_df['article_id'].unique(), self.article_encoder.transform(interactions_df['article_id'].unique())))
        self.reverse_user_mapping = {v: k for k, v in self.user_mapping.items()}
        self.reverse_article_mapping = {v: k for k, v in self.article_mapping.items()}

        user_indices = self.user_encoder.transform(interactions_df['user_id'])
        article_indices = self.article_encoder.transform(interactions_df['article_id'])
        interaction_scores = interactions_df['interaction_score'].values

        num_users = len(self.user_mapping)
        num_articles = len(self.article_mapping)
        self.model = self._create_model(num_users, num_articles, embedding_size)

        history = self.model.fit(
            [user_indices, article_indices],
            interaction_scores,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )

        self.user_factors = self.model.get_layer('user_embedding').get_weights()[0]
        self.article_factors = self.model.get_layer('article_embedding').get_weights()[0]
        self.user_biases = self.model.get_layer('user_bias').get_weights()[0]
        self.article_biases = self.model.get_layer('article_bias').get_weights()[0]
        self.global_bias = self.model.get_layer('global_bias').get_weights()[0][0]

        return self, history

    def predict(self, user_id, article_id):
        if user_id not in self.user_mapping or article_id not in self.article_mapping:
            return self.global_bias

        user_idx = self.user_mapping[user_id]
        article_idx = self.article_mapping[article_id]
        return self.model.predict([np.array([user_idx]), np.array([article_idx])])[0][0]

    def recommend_for_user(self, user_id, top_n=5, articles_df=None, exclude_seen=True, seen_interactions=None):
        if user_id not in self.user_mapping:
            return pd.DataFrame(columns=['article_id', 'score'])

        user_idx = self.user_mapping[user_id]
        user_vec = self.user_factors[user_idx]
        user_bias = self.user_biases[user_idx][0]

        scores = np.dot(self.article_factors, user_vec) + user_bias + self.article_biases.flatten() + self.global_bias
        article_ids = [self.reverse_article_mapping[i] for i in range(len(scores))]

        recommendations = pd.DataFrame({'article_id': article_ids, 'score': scores})

        if exclude_seen and seen_interactions is not None:
            seen_articles = seen_interactions[seen_interactions['user_id'] == user_id]['article_id'].values
            recommendations = recommendations[~recommendations['article_id'].isin(seen_articles)]

        recommendations = recommendations.sort_values('score', ascending=False).head(top_n)

        if articles_df is not None:
            recommendations = recommendations.merge(
                articles_df[['UUID', 'title', 'province', 'city']],
                left_on='article_id',
                right_on='UUID',
                how='inner'
            )

        return recommendations

    def recommend_similar_articles(self, article_id, top_n=5, articles_df=None):
        if article_id not in self.article_mapping:
            return pd.DataFrame(columns=['article_id', 'similarity'])

        article_idx = self.article_mapping[article_id]
        article_vec = self.article_factors[article_idx]

        similarity = np.dot(self.article_factors, article_vec) / (
            np.linalg.norm(self.article_factors, axis=1) * np.linalg.norm(article_vec)
        )

        article_ids = [self.reverse_article_mapping[i] for i in range(len(similarity))]
        similar_articles = pd.DataFrame({'article_id': article_ids, 'similarity': similarity})
        similar_articles = similar_articles[similar_articles['article_id'] != article_id]
        similar_articles = similar_articles.sort_values('similarity', ascending=False).head(top_n)

        if articles_df is not None:
            similar_articles = similar_articles.merge(
                articles_df[['UUID', 'title', 'province', 'city']],
                left_on='article_id',
                right_on='UUID',
                how='inner'
            )

        return similar_articles

    def save_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        if self.model is not None:
            self.model.save_weights(os.path.join(self.model_path, 'weights.h5'))

        if self.user_factors is not None:
            np.save(os.path.join(self.model_path, 'user_factors.npy'), self.user_factors)
        if self.article_factors is not None:
            np.save(os.path.join(self.model_path, 'article_factors.npy'), self.article_factors)
        if self.user_biases is not None:
            np.save(os.path.join(self.model_path, 'user_biases.npy'), self.user_biases)
        if self.article_biases is not None:
            np.save(os.path.join(self.model_path, 'article_biases.npy'), self.article_biases)

        with open(os.path.join(self.model_path, 'user_mapping.json'), 'w') as f:
            json.dump({str(k): int(v) for k, v in self.user_mapping.items()}, f)
        with open(os.path.join(self.model_path, 'article_mapping.json'), 'w') as f:
            json.dump({str(k): int(v) for k, v in self.article_mapping.items()}, f)
        with open(os.path.join(self.model_path, 'global_bias.json'), 'w') as f:
            json.dump(float(self.global_bias), f)

        return True

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
        self.model.load_weights(os.path.join(self.model_path, 'weights.h5'))

        return True
