from flask import Flask
from dotenv import load_dotenv
from config import Config
from extensions import db  # ⬅️ Import dari extensions
from faker import Faker
load_dotenv()

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)  # ⬅️ Gunakan init_app

# Import model setelah inisialisasi db
from models.article_model import Article

# Buat tabel
with app.app_context():
    db.create_all()
    print("✅ Tables created.")

# Lanjutkan import lainnya
from recommender_model import TFContentBasedRecommender, TFCollaborativeRecommender
from repository.article_repository import ArticleRepository
from service.recommendation_service import RecommendationService
from controller.recommendation_controller import recommendation_bp, init_routes

fake = Faker()

def generate_random_article_data(n=1000):
    articles = []
    for _ in range(n):
        article = Article(
            uuid=fake.uuid4(),
            title=fake.sentence(nb_words=6),  # Judul artikel acak
            slug=fake.slug(),
            province=fake.state(),
            city=fake.city(),
            active=True,
            user_id=fake.uuid4()
        )
        articles.append(article)
    return articles

def insert_articles():
    with app.app_context():  # Pastikan berada dalam konteks aplikasi
        articles = generate_random_article_data(1000)
        db.session.add_all(articles)
        db.session.commit()
        print(f"{len(articles)} articles inserted successfully.")
    
# Load models
content_model = TFContentBasedRecommender()
content_model.load_model()

collaborative_model = TFCollaborativeRecommender()
collaborative_model.load_model()

# Init service & repository
article_repo = ArticleRepository(content_model.articles_df)
recommendation_service = RecommendationService(content_model, collaborative_model, article_repo)

# Register routes
init_routes(recommendation_service)
app.register_blueprint(recommendation_bp)

if __name__ == '__main__':
    #insert_articles()  # Memastikan fungsi ini berjalan di dalam aplikasi context
    app.run(debug=True)
