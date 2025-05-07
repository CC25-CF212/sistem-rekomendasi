from extensions import db  # ⬅️ Ambil db dari extensions

class ArticleLike(db.Model):
    __tablename__ = 'article_likes'

    id = db.Column(db.String(36),unique=True, primary_key=True)
    user_id = db.Column(db.String(36), nullable=False)
    article_id = db.Column(db.Integer, db.ForeignKey('articles.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    article = db.relationship('Article', backref=db.backref('article_likes', lazy=True))

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "article_id": self.article_id,
            "created_at": self.created_at
        }
