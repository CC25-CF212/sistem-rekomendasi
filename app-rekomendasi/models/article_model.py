from extensions import db  # ⬅️ Ambil db dari extensions

class Article(db.Model):
    __tablename__ = 'articles'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    slug = db.Column(db.String(255), nullable=False)
    province = db.Column(db.String(100))
    city = db.Column(db.String(100))
    active = db.Column(db.Boolean, default=True)
    user_id = db.Column(db.String(36), nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "slug": self.slug,
            "province": self.province,
            "city": self.city,
            "active": self.active,
            "user_id": self.user_id
        }

