from flask import Flask, jsonify, request
from recommender_model import TFContentBasedRecommender
import pandas as pd

app = Flask(__name__)
recommender = TFContentBasedRecommender()
recommender.load_model()

# Simulasi histori baca
user_history = {
    "user_1": "a1b2c3d4-1111"
}

# Endpoint untuk simpan artikel yang dibaca
@app.route('/api/read', methods=['POST'])
def save_article_read():
    data = request.json
    user_id = data.get("user_id")
    article_id = data.get("article_id")

    if not user_id or not article_id:
        return jsonify({"error": "user_id and article_id required"}), 400

    user_history[user_id] = article_id
    return jsonify({"message": "Read article saved"})

# Endpoint rekomendasi
@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    # user_id = request.args.get('user_id')
    user_id = "a1b2c3d4-1111"
    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    last_article = user_history.get(user_id)

    if not last_article:
        # Kalau belum pernah baca, kembalikan 5 artikel pertama
        default_articles = recommender.articles_df.head(5)
        return jsonify(default_articles[['UUID', 'title','province','city']].to_dict(orient='records'))

    recommended = recommender.recommend(article_id=last_article, top_n=5)
    return jsonify(recommended)

if __name__ == '__main__':
    app.run(debug=True)
