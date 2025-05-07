from extensions import db  # ⬅️ Ambil db dari extensions


class ArticleComment(db.Model):
    __tablename__ = 'article_comments'

    id = db.Column(db.String(36),unique=True, primary_key=True)
    comment = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    parent_comment_id = db.Column(db.String(36), db.ForeignKey('article_comments.id'))
    article_id = db.Column(db.Integer, db.ForeignKey('articles.id'), nullable=False)
    user_id = db.Column(db.String(36), nullable=False)

    parent_comment = db.relationship('ArticleComment', remote_side=[id], backref=db.backref('replies', lazy=True))
    article = db.relationship('Article', backref=db.backref('article_comments', lazy=True))

    def to_dict(self):
        return {
            "id": self.id,
            "comment": self.comment,
            "created_at": self.created_at,
            "parent_comment_id": self.parent_comment_id,
            "article_id": self.article_id,
            "user_id": self.user_id
        }