class RecommendationService:
    def __init__(self, content_recommender, collaborative_recommender, article_repository):
        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.article_repository = article_repository
        self.user_history = {
            "user_1": "a1b2c3d4-1111"
        }

    def get_collaborative_recommendations(self, user_id):
        if not user_id:
            raise ValueError("user_id is required")

        last_article = self.user_history.get(user_id)
        if not last_article:
            return self.article_repository.get_default_articles().to_dict(orient='records')

        return self.content_recommender.recommend(article_id=last_article, top_n=5)

    def get_content_based_recommendations(self, user_id):
        if not user_id:
            raise ValueError("user_id is required")
        return [
            {'article_id': article_id, 'score': float(score)}
            for article_id, score in self.collaborative_recommender.recommend_for_user(user_id, top_n=5)
        ]
    def get_article_by_id(self, article_id):
        # Menggunakan method repository untuk mendapatkan artikel berdasarkan ID
        article = self.article_repository.get_by_id(article_id)
        if not article:
            return {"message": "Article not found"}, 404
        return article.to_dict(), 200
    
    def get_all_articles(self):
        # Menggunakan method repository untuk mendapatkan semua artikel
        articles = self.article_repository.get_all()
        if not articles:
            return {"message": "No articles found"}, 404

        # Mengonversi daftar artikel ke format dictionary
        return [article.to_dict() for article in articles], 200