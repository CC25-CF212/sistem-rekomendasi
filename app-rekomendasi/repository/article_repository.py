import pandas as pd
from models.article_model import Article
class ArticleRepository:
    def __init__(self, articles_df):
        self.articles_df = articles_df

    def get_default_articles(self, limit=5):
        return self.articles_df.head(limit)[['UUID', 'title', 'province', 'city']]
    
    @staticmethod
    def get_by_id(article_id):
        return Article.query.get(article_id)

    @staticmethod
    def get_all():
        return Article.query.all()