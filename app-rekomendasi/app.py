from flask import Flask, jsonify, request
from recommender_model import TFContentBasedRecommender,TFCollaborativeRecommender
import pandas as pd

app = Flask(__name__)
recommenderv1 = TFContentBasedRecommender()
recommenderv1.load_model()

recommenderv2 = TFCollaborativeRecommender()
recommenderv2.load_model()

# Simulasi histori baca
user_history = {
    "user_1": "a1b2c3d4-1111"
}


@app.route('/api/recommendations/collaborative', methods=['GET'])
def get_recommendations():
    user_id = "1"
    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    last_article = user_history.get(user_id)

    if not last_article:
        default_articles = recommenderv1.articles_df.head(5)
        return jsonify(default_articles[['UUID', 'title','province','city']].to_dict(orient='records'))

    recommended = recommenderv1.recommend(article_id=last_article, top_n=5)
    return jsonify(recommended)

@app.route('/api/recommendations/content-based-recommender', methods=['GET'])
def recommend():
    user_id = "u1e2f3g4-2222"
    if not user_id:
        return jsonify({'error': 'Parameter user_id diperlukan'}), 400

    try:
        recommendations = recommenderv2.recommend_for_user(user_id, top_n=5)
        return jsonify([
            {'article_id': article_id, 'score': float(score)}
            for article_id, score in recommendations
        ])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
