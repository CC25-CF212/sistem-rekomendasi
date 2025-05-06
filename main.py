import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pickle
import json
import re
import os
from datetime import datetime

# ===============================
# 1. LOAD & PREPROCESSING DATA
# ===============================

# Function to load data
def load_data():
    # Dalam implementasi nyata, Anda akan mengambil data dari database
    # Untuk contoh ini, kita gunakan data hardcoded sebagai DataFrame
    
    # Articles data
    articles_data = [
        {'ID': 1, 'UUID': 'a1b2c3d4-1111', 'title': 'Wisata Pantai Kuta', 'slug': 'wisata-pantai-kuta', 
         'province': 'Bali', 'city': 'Denpasar', 'active': True, 'User_ID': 'u1e2f3g4-1111'},
        {'ID': 2, 'UUID': 'a1b2c3d4-2222', 'title': 'Kuliner Gudeg Jogja', 'slug': 'kuliner-gudeg-jogja', 
         'province': 'DI Yogyakarta', 'city': 'Yogyakarta', 'active': True, 'User_ID': 'u1e2f3g4-2222'},
        {'ID': 3, 'UUID': 'a1b2c3d4-3333', 'title': 'Mendaki Gunung Bromo', 'slug': 'mendaki-gunung-bromo', 
         'province': 'Jawa Timur', 'city': 'Probolinggo', 'active': True, 'User_ID': 'u1e2f3g4-1111'},
        {'ID': 4, 'UUID': 'a1b2c3d4-4444', 'title': 'Sejarah Candi Borobudur', 'slug': 'sejarah-candi-borobudur', 
         'province': 'Jawa Tengah', 'city': 'Magelang', 'active': True, 'User_ID': 'u1e2f3g4-3333'},
        {'ID': 5, 'UUID': 'a1b2c3d4-5555', 'title': 'Tips Belanja di Malioboro', 'slug': 'tips-belanja-malioboro', 
         'province': 'DI Yogyakarta', 'city': 'Yogyakarta', 'active': True, 'User_ID': 'u1e2f3g4-2222'},
        {'ID': 6, 'UUID': 'a1b2c3d4-6666', 'title': 'Mengenal Tari Kecak', 'slug': 'mengenal-tari-kecak', 
         'province': 'Bali', 'city': 'Denpasar', 'active': True, 'User_ID': 'u1e2f3g4-4444'},
        {'ID': 7, 'UUID': 'a1b2c3d4-7777', 'title': 'Keindahan Raja Ampat', 'slug': 'keindahan-raja-ampat', 
         'province': 'Papua Barat', 'city': 'Raja Ampat', 'active': True, 'User_ID': 'u1e2f3g4-3333'},
        {'ID': 8, 'UUID': 'a1b2c3d4-8888', 'title': 'Menjelajah Kawah Ijen', 'slug': 'menjelajah-kawah-ijen', 
         'province': 'Jawa Timur', 'city': 'Banyuwangi', 'active': True, 'User_ID': 'u1e2f3g4-1111'},
        {'ID': 9, 'UUID': 'a1b2c3d4-9999', 'title': 'Festival Cap Go Meh', 'slug': 'festival-cap-go-meh', 
         'province': 'Kalimantan Barat', 'city': 'Singkawang', 'active': True, 'User_ID': 'u1e2f3g4-5555'},
        {'ID': 10, 'UUID': 'a1b2c3d4-0000', 'title': 'Tradisi Ngaben di Bali', 'slug': 'tradisi-ngaben-bali', 
         'province': 'Bali', 'city': 'Denpasar', 'active': True, 'User_ID': 'u1e2f3g4-4444'}
    ]
    
    # Article likes data
    likes_data = [
        {'ID': 1, 'UUID': 'l1m2n3o4-1111', 'user_id': 'u1e2f3g4-3333', 'article_id': 'a1b2c3d4-1111'},
        {'ID': 2, 'UUID': 'l1m2n3o4-2222', 'user_id': 'u1e2f3g4-5555', 'article_id': 'a1b2c3d4-1111'},
        {'ID': 3, 'UUID': 'l1m2n3o4-3333', 'user_id': 'u1e2f3g4-2222', 'article_id': 'a1b2c3d4-1111'},
        {'ID': 4, 'UUID': 'l1m2n3o4-4444', 'user_id': 'u1e2f3g4-1111', 'article_id': 'a1b2c3d4-2222'},
        {'ID': 5, 'UUID': 'l1m2n3o4-5555', 'user_id': 'u1e2f3g4-3333', 'article_id': 'a1b2c3d4-2222'},
        {'ID': 6, 'UUID': 'l1m2n3o4-6666', 'user_id': 'u1e2f3g4-4444', 'article_id': 'a1b2c3d4-3333'},
        {'ID': 7, 'UUID': 'l1m2n3o4-7777', 'user_id': 'u1e2f3g4-5555', 'article_id': 'a1b2c3d4-3333'},
        {'ID': 8, 'UUID': 'l1m2n3o4-8888', 'user_id': 'u1e2f3g4-2222', 'article_id': 'a1b2c3d4-3333'},
        {'ID': 9, 'UUID': 'l1m2n3o4-9999', 'user_id': 'u1e2f3g4-1111', 'article_id': 'a1b2c3d4-3333'},
        {'ID': 10, 'UUID': 'l1m2n3o4-0001', 'user_id': 'u1e2f3g4-4444', 'article_id': 'a1b2c3d4-4444'},
        {'ID': 11, 'UUID': 'l1m2n3o4-0002', 'user_id': 'u1e2f3g4-3333', 'article_id': 'a1b2c3d4-5555'},
        {'ID': 12, 'UUID': 'l1m2n3o4-0003', 'user_id': 'u1e2f3g4-1111', 'article_id': 'a1b2c3d4-5555'},
        {'ID': 13, 'UUID': 'l1m2n3o4-0004', 'user_id': 'u1e2f3g4-5555', 'article_id': 'a1b2c3d4-6666'},
        {'ID': 14, 'UUID': 'l1m2n3o4-0005', 'user_id': 'u1e2f3g4-2222', 'article_id': 'a1b2c3d4-7777'},
        {'ID': 15, 'UUID': 'l1m2n3o4-0006', 'user_id': 'u1e2f3g4-3333', 'article_id': 'a1b2c3d4-7777'},
        {'ID': 16, 'UUID': 'l1m2n3o4-0007', 'user_id': 'u1e2f3g4-4444', 'article_id': 'a1b2c3d4-8888'},
        {'ID': 17, 'UUID': 'l1m2n3o4-0008', 'user_id': 'u1e2f3g4-1111', 'article_id': 'a1b2c3d4-8888'},
        {'ID': 18, 'UUID': 'l1m2n3o4-0009', 'user_id': 'u1e2f3g4-2222', 'article_id': 'a1b2c3d4-9999'},
        {'ID': 19, 'UUID': 'l1m2n3o4-0010', 'user_id': 'u1e2f3g4-4444', 'article_id': 'a1b2c3d4-9999'},
        {'ID': 20, 'UUID': 'l1m2n3o4-0011', 'user_id': 'u1e2f3g4-3333', 'article_id': 'a1b2c3d4-0000'}
    ]
    
    # Article comments data
    comments_data = [
        {'ID': 1, 'UUID': 'c1d2e3f4-1111', 'article_id': 'a1b2c3d4-1111', 'user_id': 'u1e2f3g4-3333'},
        {'ID': 2, 'UUID': 'c1d2e3f4-2222', 'article_id': 'a1b2c3d4-1111', 'user_id': 'u1e2f3g4-5555'},
        {'ID': 3, 'UUID': 'c1d2e3f4-3333', 'article_id': 'a1b2c3d4-1111', 'user_id': 'u1e2f3g4-2222'},
        {'ID': 4, 'UUID': 'c1d2e3f4-4444', 'article_id': 'a1b2c3d4-1111', 'user_id': 'u1e2f3g4-1111'},
        {'ID': 5, 'UUID': 'c1d2e3f4-5555', 'article_id': 'a1b2c3d4-2222', 'user_id': 'u1e2f3g4-1111'},
        {'ID': 6, 'UUID': 'c1d2e3f4-6666', 'article_id': 'a1b2c3d4-2222', 'user_id': 'u1e2f3g4-3333'},
        {'ID': 7, 'UUID': 'c1d2e3f4-7777', 'article_id': 'a1b2c3d4-2222', 'user_id': 'u1e2f3g4-2222'},
        {'ID': 8, 'UUID': 'c1d2e3f4-8888', 'article_id': 'a1b2c3d4-3333', 'user_id': 'u1e2f3g4-4444'},
        {'ID': 9, 'UUID': 'c1d2e3f4-9999', 'article_id': 'a1b2c3d4-3333', 'user_id': 'u1e2f3g4-5555'},
        {'ID': 10, 'UUID': 'c1d2e3f4-0001', 'article_id': 'a1b2c3d4-3333', 'user_id': 'u1e2f3g4-1111'},
        {'ID': 11, 'UUID': 'c1d2e3f4-0002', 'article_id': 'a1b2c3d4-4444', 'user_id': 'u1e2f3g4-4444'},
        {'ID': 12, 'UUID': 'c1d2e3f4-0003', 'article_id': 'a1b2c3d4-5555', 'user_id': 'u1e2f3g4-3333'},
        {'ID': 13, 'UUID': 'c1d2e3f4-0004', 'article_id': 'a1b2c3d4-5555', 'user_id': 'u1e2f3g4-1111'},
        {'ID': 14, 'UUID': 'c1d2e3f4-0005', 'article_id': 'a1b2c3d4-5555', 'user_id': 'u1e2f3g4-2222'},
        {'ID': 15, 'UUID': 'c1d2e3f4-0006', 'article_id': 'a1b2c3d4-6666', 'user_id': 'u1e2f3g4-5555'},
        {'ID': 16, 'UUID': 'c1d2e3f4-0007', 'article_id': 'a1b2c3d4-7777', 'user_id': 'u1e2f3g4-2222'},
        {'ID': 17, 'UUID': 'c1d2e3f4-0008', 'article_id': 'a1b2c3d4-8888', 'user_id': 'u1e2f3g4-4444'},
        {'ID': 18, 'UUID': 'c1d2e3f4-0009', 'article_id': 'a1b2c3d4-9999', 'user_id': 'u1e2f3g4-2222'},
        {'ID': 19, 'UUID': 'c1d2e3f4-0010', 'article_id': 'a1b2c3d4-0000', 'user_id': 'u1e2f3g4-3333'},
        {'ID': 20, 'UUID': 'c1d2e3f4-0011', 'article_id': 'a1b2c3d4-0000', 'user_id': 'u1e2f3g4-5555'}
    ]
    
    # Konversi ke DataFrame
    articles_df = pd.DataFrame(articles_data)
    likes_df = pd.DataFrame(likes_data)
    comments_df = pd.DataFrame(comments_data)
    
    return articles_df, likes_df, comments_df

# Function to preprocess data
def preprocess_data(articles_df, likes_df, comments_df):
    # Hitung jumlah likes per artikel
    article_likes = likes_df.groupby('article_id').size().reset_index(name='likes_count')
    
    # Hitung jumlah comments per artikel
    article_comments = comments_df.groupby('article_id').size().reset_index(name='comments_count')
    
    # Gabungkan data dengan artikel
    articles_enriched = articles_df.copy()
    articles_enriched['UUID'] = articles_enriched['UUID'].astype(str)
    
    # Tambahkan likes count
    articles_enriched = articles_enriched.merge(article_likes, left_on='UUID', right_on='article_id', how='left')
    articles_enriched['likes_count'] = articles_enriched['likes_count'].fillna(0)
    
    # Tambahkan comments count
    articles_enriched = articles_enriched.merge(article_comments, left_on='UUID', right_on='article_id', how='left')
    articles_enriched['comments_count'] = articles_enriched['comments_count'].fillna(0)
    
    # Clean up
    articles_enriched = articles_enriched.drop(['article_id_x', 'article_id_y'], axis=1, errors='ignore')
    
    # Ekstrak fitur-fitur dari judul artikel menggunakan Text Processing
    # Menggabungkan title, province, dan city untuk text-based features
    articles_enriched['text_features'] = articles_enriched['title'] + ' ' + articles_enriched['province'] + ' ' + articles_enriched['city']
    
    # Tambahkan fitur engagement
    articles_enriched['engagement_score'] = articles_enriched['likes_count'] + (2 * articles_enriched['comments_count'])
    
    return articles_enriched

# ===============================
# 2. MODEL DEVELOPMENT
# ===============================

# MODEL 1: Content-Based Filtering dengan TF-IDF
class ContentBasedRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.articles_df = None
        self.model_path = 'models/content_based_recommender.pkl'
    
    def fit(self, articles_df):
        self.articles_df = articles_df.copy()
        self.tfidf_matrix = self.vectorizer.fit_transform(articles_df['text_features'])
        return self
    
    def recommend(self, article_id, top_n=5):
        # Find article index
        idx = self.articles_df[self.articles_df['UUID'] == article_id].index[0]
        
        # Compute similarity scores
        sim_scores = cosine_similarity(self.tfidf_matrix[idx:idx+1], self.tfidf_matrix).flatten()
        
        # Get top similar articles (excluding itself)
        sim_indices = sim_scores.argsort()[::-1][1:top_n+1]
        
        recommended_articles = self.articles_df.iloc[sim_indices][['UUID', 'title', 'province', 'city']].copy()
        recommended_articles['similarity_score'] = sim_scores[sim_indices]
        
        return recommended_articles
    
    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'articles_df': self.articles_df
            }, f)
    
    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.vectorizer = model_data['vectorizer']
                self.tfidf_matrix = model_data['tfidf_matrix']
                self.articles_df = model_data['articles_df']
            return True
        return False
    
    def update_model(self, new_articles_df):
        # Combine existing and new articles
        if self.articles_df is not None:
            # Find new articles that aren't in the existing model
            existing_uuids = set(self.articles_df['UUID'].values)
            new_articles = new_articles_df[~new_articles_df['UUID'].isin(existing_uuids)]
            
            if len(new_articles) > 0:
                # Update dataframe with new articles
                self.articles_df = pd.concat([self.articles_df, new_articles]).reset_index(drop=True)
                
                # Recompute TF-IDF matrix
                self.tfidf_matrix = self.vectorizer.fit_transform(self.articles_df['text_features'])
                
                # Save updated model
                self.save_model()
                
                return True, f"Model updated with {len(new_articles)} new articles"
            
            return False, "No new articles to update"
        else:
            # If model is empty, just fit with new data
            self.fit(new_articles_df)
            self.save_model()
            return True, f"Initial model created with {len(new_articles_df)} articles"

# MODEL 2: Collaborative Filtering based on User Interactions
class CollaborativeRecommender:
    def __init__(self):
        self.user_article_matrix = None
        self.articles_df = None
        self.model_path = 'models/collaborative_recommender.pkl'
    
    def fit(self, articles_df, likes_df, comments_df):
        self.articles_df = articles_df.copy()
        
        # Create user-article interaction matrix
        # Combine likes and comments with weighted values
        likes_df['interaction'] = 1.0  # Weight for likes
        comments_df['interaction'] = 2.0  # Weight for comments (comments count more)
        
        # Rename columns for consistency
        likes_df = likes_df.rename(columns={'article_id': 'article_id', 'user_id': 'user_id'})
        comments_df = comments_df.rename(columns={'article_id': 'article_id', 'user_id': 'user_id'})
        
        # Combine interactions
        interactions = pd.concat([
            likes_df[['user_id', 'article_id', 'interaction']],
            comments_df[['user_id', 'article_id', 'interaction']]
        ])
        
        # Sum interactions per user-article pair
        interactions = interactions.groupby(['user_id', 'article_id']).sum().reset_index()
        
        # Create the user-article matrix
        self.user_article_matrix = interactions.pivot_table(
            index='user_id', 
            columns='article_id', 
            values='interaction',
            fill_value=0
        )
        
        return self
    
    def recommend(self, user_id, top_n=5):
        if user_id not in self.user_article_matrix.index:
            # Cold start problem - return popular articles
            popular_articles = self.articles_df.sort_values('likes_count', ascending=False).head(top_n)
            return popular_articles[['UUID', 'title', 'province', 'city']].copy()
        
        # Get user's interactions
        user_interactions = self.user_article_matrix.loc[user_id]
        
        # Get articles the user hasn't interacted with
        non_interacted_articles = user_interactions[user_interactions == 0].index.tolist()
        
        # Calculate similarity between this user and all others
        user_similarity = cosine_similarity(
            self.user_article_matrix.loc[[user_id]],
            self.user_article_matrix
        ).flatten()
        
        # Get most similar users
        similar_user_indices = user_similarity.argsort()[::-1][1:6]  # Top 5 similar users
        similar_users = self.user_article_matrix.iloc[similar_user_indices]
        
        # Get recommendations from similar users
        recommendations = {}
        
        for article_id in non_interacted_articles:
            if article_id in similar_users.columns:
                article_ratings = similar_users[article_id]
                weighted_score = sum(article_ratings * user_similarity[similar_user_indices]) / sum(user_similarity[similar_user_indices])
                recommendations[article_id] = weighted_score
        
        # Sort recommendations
        recommended_article_ids = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
        recommended_article_ids = [article_id for article_id, score in recommended_article_ids]
        
        # Get article details
        recommended_articles = self.articles_df[self.articles_df['UUID'].isin(recommended_article_ids)][['UUID', 'title', 'province', 'city']].copy()
        
        return recommended_articles
    
    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'user_article_matrix': self.user_article_matrix,
                'articles_df': self.articles_df
            }, f)
    
    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.user_article_matrix = model_data['user_article_matrix']
                self.articles_df = model_data['articles_df']
            return True
        return False
    
    def update_model(self, new_articles_df, new_likes_df, new_comments_df):
        if self.user_article_matrix is not None:
            # Combine existing and new articles
            existing_uuids = set(self.articles_df['UUID'].values)
            new_articles = new_articles_df[~new_articles_df['UUID'].isin(existing_uuids)]
            
            # Combine articles
            self.articles_df = pd.concat([self.articles_df, new_articles]).reset_index(drop=True)
            
            # Update interactions with new data
            self.fit(self.articles_df, new_likes_df, new_comments_df)
            
            # Save updated model
            self.save_model()
            
            return True, f"Model updated with {len(new_articles)} new articles and new interactions"
        else:
            # If model is empty, just fit with new data
            self.fit(new_articles_df, new_likes_df, new_comments_df)
            self.save_model()
            return True, f"Initial model created with {len(new_articles_df)} articles"

# MODEL 3: Hybrid Recommender (Combines Content-Based and Collaborative Filtering)
class HybridRecommender:
    def __init__(self):
        self.content_recommender = ContentBasedRecommender()
        self.collab_recommender = CollaborativeRecommender()
        self.articles_df = None
        self.model_path = 'models/hybrid_recommender.pkl'
    
    def fit(self, articles_df, likes_df, comments_df):
        self.articles_df = articles_df.copy()
        self.content_recommender.fit(articles_df)
        self.collab_recommender.fit(articles_df, likes_df, comments_df)
        return self
    
    def recommend(self, user_id, article_id=None, top_n=5, content_weight=0.5):
        # Get content-based recommendations
        if article_id:
            content_recs = self.content_recommender.recommend(article_id, top_n=top_n)
            content_article_ids = content_recs['UUID'].tolist()
            content_scores = dict(zip(content_article_ids, content_recs['similarity_score'].values))
        else:
            # If no article_id, use random article from user history
            user_likes = self.collab_recommender.user_article_matrix.loc[user_id]
            interacted_articles = user_likes[user_likes > 0].index.tolist()
            
            if interacted_articles:
                random_article = np.random.choice(interacted_articles)
                content_recs = self.content_recommender.recommend(random_article, top_n=top_n)
                content_article_ids = content_recs['UUID'].tolist()
                content_scores = dict(zip(content_article_ids, content_recs['similarity_score'].values))
            else:
                content_article_ids = []
                content_scores = {}
        
        # Get collaborative filtering recommendations
        collab_recs = self.collab_recommender.recommend(user_id, top_n=top_n)
        collab_article_ids = collab_recs['UUID'].tolist()
        
        # Combine recommendations
        all_recommendations = set(content_article_ids + collab_article_ids)
        
        # Scoring
        hybrid_scores = {}
        max_content_score = max(content_scores.values()) if content_scores else 1
        
        for article_id in all_recommendations:
            # Normalize content score
            content_score = content_scores.get(article_id, 0) / max_content_score if max_content_score > 0 else 0
            
            # Calculate collaborative score (binary for now - just if it's in the collab recommendations)
            collab_score = 1.0 if article_id in collab_article_ids else 0.0
            
            # Weighted hybrid score
            hybrid_scores[article_id] = (content_weight * content_score) + ((1 - content_weight) * collab_score)
        
        # Get top hybrid recommendations
        top_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_article_ids = [article_id for article_id, _ in top_hybrid]
        
        # Get article details
        recommended_articles = self.articles_df[self.articles_df['UUID'].isin(top_article_ids)][['UUID', 'title', 'province', 'city']].copy()
        
        # Add scores
        recommended_articles['hybrid_score'] = recommended_articles['UUID'].map(dict(top_hybrid))
        
        # Sort by score
        recommended_articles = recommended_articles.sort_values('hybrid_score', ascending=False)
        
        return recommended_articles
    
    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save individual models first
        self.content_recommender.save_model()
        self.collab_recommender.save_model()
        
        # Save hybrid model info
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'articles_df': self.articles_df
            }, f)
    
    # def load_model(self):
    #     if os.path.exists(self.model_path):
    #         # Load individual models
    #         content_loaded = self.content_recommender.load_model()
    #         collab_loaded = self.collab_recommender.load_model()
            
    #          # Load hybrid model data
    #         with open(self.model_path, 'rb') as f:
    #             model_data = pickle.load(f)
    #             self.articles_df = model_data['articles_df']
            
    #         return content_loaded and collab_loaded
    #     else:
    #         return False
        
    def load_model(self):
        print("HybridRecommender: Loading model components...")
        content_loaded = self.content_recommender.load_model()
        collab_loaded = self.collab_recommender.load_model()
        
        hybrid_state_loaded = False
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.articles_df = model_data['articles_df']
                print(f"HybridRecommender: Own state (articles_df) loaded from {self.model_path}")
                hybrid_state_loaded = True
            except Exception as e:
                print(f"HybridRecommender: Error loading own state - {e}")
                self.articles_df = None # Ensure it's reset on failure
        else:
            print(f"HybridRecommender: Own state file {self.model_path} not found.")

        if content_loaded and collab_loaded and hybrid_state_loaded:
            # Verify consistency if needed, e.g., if articles_df in sub-models should match hybrid's
            # For now, assume they are consistent if all loaded successfully from a proper save.
            print("HybridRecommender: All components loaded successfully.")
            return True
        else:
            print("HybridRecommender: Failed to load one or more components. Model may be in an inconsistent state.")
            return False
    
    def update_model(self, raw_articles_complete_df, raw_likes_complete_df, raw_comments_complete_df):
        """
        Updates the hybrid recommender and its sub-models with new (complete) data.
        The input dataframes are expected to be the complete, latest versions of raw data.
        """
        print("\nHybridRecommender: Starting model update process...")

        # 1. Preprocess the complete raw data to get the latest enriched articles dataframe
        print("HybridRecommender: Preprocessing complete dataset for update...")
        current_articles_enriched_df = preprocess_data(
            raw_articles_complete_df,
            raw_likes_complete_df,
            raw_comments_complete_df
        )
        self.articles_df = current_articles_enriched_df.copy() # Update master articles_df
        print(f"HybridRecommender: Preprocessing for update complete. Enriched articles count: {len(self.articles_df)}")

        # Ensure sub-models are loaded or initialized if they weren't already
        # (Their update_model methods also handle initial fitting if they are empty)
        if self.content_recommender.articles_df is None and not os.path.exists(self.content_recommender.model_path):
            print("HybridRecommender: Content-Based model seems uninitialized and no saved file. Will be fitted by its update.")
        
        if self.collab_recommender.user_article_matrix is None and not os.path.exists(self.collab_recommender.model_path):
            print("HybridRecommender: Collaborative model seems uninitialized and no saved file. Will be fitted by its update.")

        # 2. Update Content-Based Recommender
        print("\nHybridRecommender: Updating Content-Based Recommender...")
        content_updated_status, content_update_message = self.content_recommender.update_model(
            self.articles_df # Pass the full, latest enriched article set
        )
        print(f"HybridRecommender (Content-Based status): {content_update_message}")

        # 3. Update Collaborative Recommender
        print("\nHybridRecommender: Updating Collaborative Recommender...")
        collab_updated_status, collab_update_message = self.collab_recommender.update_model(
            self.articles_df,          # Full enriched articles from hybrid's perspective
            raw_likes_complete_df,    # Full raw likes
            raw_comments_complete_df  # Full raw comments
        )
        print(f"HybridRecommender (Collaborative status): {collab_update_message}")

        # Sub-models save themselves if updated. HybridRecommender saves its own state and coordinates.
        if content_updated_status or collab_updated_status:
            print("\nHybridRecommender: At least one sub-model was updated. Saving overall Hybrid model state.")
            self.save_model() # Saves hybrid's articles_df and calls save on sub-models
            final_status_message = f"Hybrid model update process complete. CB: {content_update_message} | Collab: {collab_update_message}"
            print(final_status_message)
            return True, final_status_message
        else:
            # Even if sub-models reported no changes, the hybrid's articles_df was updated.
            # So, save the hybrid model's state.
            print("\nHybridRecommender: Sub-models reported no structural changes, but saving hybrid state (e.g., updated article counts).")
            self.save_model()
            final_status_message = f"Hybrid model: No structural updates to sub-models, but state refreshed. CB: {content_update_message} | Collab: {collab_update_message}"
            print(final_status_message)
            return False, final_status_message # False indicates sub-models didn't undergo major retraining

# ===============================
# 3. MAIN EXECUTION & DEMO
# ===============================
if __name__ == "__main__":
    # --- Initial Data Load and Model Training ---
    print("--- Initial Model Training Phase ---")
    raw_articles_df, raw_likes_df, raw_comments_df = load_data()
    print(f"Loaded initial data: {len(raw_articles_df)} articles, {len(raw_likes_df)} likes, {len(raw_comments_df)} comments.")

    enriched_articles_df = preprocess_data(raw_articles_df, raw_likes_df, raw_comments_df)
    print(f"Initial data preprocessed. Enriched articles: {len(enriched_articles_df)}")
    # print("Enriched articles columns:", enriched_articles_df.columns)
    # print(enriched_articles_df[['UUID', 'title', 'likes_count', 'comments_count', 'engagement_score']].head())


    hybrid_recommender = HybridRecommender()
    hybrid_recommender.fit(enriched_articles_df, raw_likes_df, raw_comments_df)
    hybrid_recommender.save_model()
    print("--- Initial Model Training and Saving Complete ---")

    # --- Simulate a new run: Load models ---
    print("\n--- Model Loading Phase (Simulating New Run) ---")
    loaded_hybrid_recommender = HybridRecommender()
    if loaded_hybrid_recommender.load_model():
        print("Hybrid Recommender loaded successfully for inference.")
        
        # --- Get Recommendations (After Initial Load) ---
        print("\n--- Getting Recommendations (After Initial Load) ---")
        test_user_id = 'u1e2f3g4-3333' # A user from the dataset
        test_article_context_id = 'a1b2c3d4-1111' # An article for content context

        print(f"\nRecommendations for user '{test_user_id}' (no specific article context):")
        recommendations1 = loaded_hybrid_recommender.recommend(user_id=test_user_id, top_n=3)
        print(recommendations1)

        print(f"\nRecommendations for user '{test_user_id}' (with article context '{test_article_context_id}'):")
        recommendations2 = loaded_hybrid_recommender.recommend(user_id=test_user_id, article_id_context=test_article_context_id, top_n=3)
        print(recommendations2)
        
        # Test cold start user
        cold_start_user_id = 'user-new-cold-start'
        print(f"\nRecommendations for a new user '{cold_start_user_id}' (cold start):")
        recommendations_cold_start = loaded_hybrid_recommender.recommend(user_id=cold_start_user_id, top_n=3)
        print(recommendations_cold_start)

    else:
        print("Failed to load hybrid recommender. Skipping further demonstration.")
        exit()

    # --- Simulate New Data Arriving and Model Update ---
    print("\n--- Model Update Phase (Simulating New Data) ---")
    # current_raw_articles_df, current_raw_likes_df, current_raw_comments_df are the original "full" datasets
    # before this update. We use them as base for load_updated_data.
    updated_raw_articles_df, updated_raw_likes_df, updated_raw_comments_df = load_data(
        raw_articles_df, raw_likes_df, raw_comments_df
    )
    print(f"Simulated new data: Now {len(updated_raw_articles_df)} total articles, "
          f"{len(updated_raw_likes_df)} total likes, {len(updated_raw_comments_df)} total comments.")

    # Use the already loaded_hybrid_recommender instance for update
    status, message = loaded_hybrid_recommender.update_model(
        updated_raw_articles_df, 
        updated_raw_likes_df, 
        updated_raw_comments_df
    )
    print(f"Model update status: {status} - {message}")
    print("--- Model Update Complete ---")

    # --- Get Recommendations (After Model Update) ---
    print("\n--- Getting Recommendations (After Model Update) ---")
    # Test with the same user, expect potentially different recommendations
    print(f"\nRecommendations for user '{test_user_id}' after model update:")
    recommendations_after_update = loaded_hybrid_recommender.recommend(user_id=test_user_id, top_n=3)
    print(recommendations_after_update)

    # Test if new articles can be recommended (e.g., via content or popular if collab picks them up)
    # Example: recommendations for user 'u1e2f3g4-1111' who created one of the new articles.
    # The new article 'a1b2c3d4-AAAA' (Resep Nasi Goreng Spesial Jawa)
    print(f"\nRecommendations for user 'u1e2f3g4-1111' (author of a new article) after update, "
          f"using context of new article 'a1b2c3d4-AAAA':")
    recs_for_new_article_author = loaded_hybrid_recommender.recommend(user_id='u1e2f3g4-1111', article_id_context='a1b2c3d4-AAAA', top_n=3)
    print(recs_for_new_article_author)

    print("\n--- Demonstration Finished ---")