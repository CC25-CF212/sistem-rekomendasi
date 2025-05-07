# from flask import Blueprint, request, jsonify

# recommendation_bp = Blueprint('recommendation', __name__)

# def init_routes(service):

#     @recommendation_bp.route('/api/recommendations/collaborative', methods=['GET'])
#     def get_collaborative():
#         user_id = request.args.get('user_id', 'user_1')
#         print(f"User ID: {user_id}")  # Debugging line to check user_id
#         try:
#             recommendations = service.get_collaborative_recommendations(user_id)
#             return jsonify({
#                 "status": "success",
#                 "message": "Collaborative recommendations fetched",
#                 "data": recommendations
#             }), 200
#         except ValueError as ve:
#             return jsonify({"status": "error", "message": str(ve)}), 400
#         except Exception as e:
#             return jsonify({"status": "error", "message": "Internal server error", "details": str(e)}), 500

#     @recommendation_bp.route('/api/recommendations/content-based-recommender', methods=['GET'])
#     def get_content_based():
#         user_id = request.args.get('user_id', 'u1e2f3g4-2222')
#         try:
#             recommendations = service.get_content_based_recommendations(user_id)
#             return jsonify({
#                 "status": "success",
#                 "message": "Content-based recommendations fetched",
#                 "data": recommendations
#             }), 200
#         except ValueError as ve:
#             return jsonify({"status": "error", "message": str(ve)}), 400
#         except Exception as e:
#             return jsonify({"status": "error", "message": "Internal server error", "details": str(e)}), 500

import random
from flask import Blueprint, request
from utils.response_builder import ResponseBuilder
from extensions import db
from datetime import datetime
import uuid
from models.article_like_model import ArticleLike
from models.article_comment_model import ArticleComment
recommendation_bp = Blueprint('recommendation', __name__)

def init_routes(service):

    @recommendation_bp.route('/api/recommendations/collaborative', methods=['GET'])
    def get_collaborative():
        user_id = request.args.get('user_id', 'user_1')
        try:
            recommendations = service.get_collaborative_recommendations(user_id)
            return ResponseBuilder.success("Collaborative recommendations fetched", recommendations)
        except ValueError as ve:
            return ResponseBuilder.error(str(ve), status_code=400)
        except Exception as e:
            return ResponseBuilder.error("Internal server error", details=str(e), status_code=500)

    @recommendation_bp.route('/api/recommendations/content-based-recommender', methods=['GET'])
    def get_content_based():
        user_id = request.args.get('user_id', 'u1e2f3g4-2222')
        try:
            recommendations = service.get_content_based_recommendations(user_id)
            return ResponseBuilder.success("Content-based recommendations fetched", recommendations)
        except ValueError as ve:
            return ResponseBuilder.error(str(ve), status_code=400)
        except Exception as e:
            return ResponseBuilder.error("Internal server error", details=str(e), status_code=500)

    @recommendation_bp.route('/api/articles', methods=['GET'])
    def get_articles():
        # Memanggil service untuk mendapatkan semua artikel
        articles, status_code = service.get_all_articles()
        # Insert data ke tabel likes dan comments untuk setiap artikel
        # Daftar ID user yang diizinkan (misalnya 5 user tetap)
        user_ids = [1, 2, 3, 4, 5]
        for article_data in articles:
            # Pilih user secara acak dari daftar
            random_user_id = random.choice(user_ids)
    
            # Insert Like
            like = ArticleLike(
                id=str(uuid.uuid4()),
                user_id=random_user_id,
                article_id=article_data['id'],
                created_at=datetime.utcnow()
            )
            db.session.add(like)

            # Insert Comment
            comment = ArticleComment(
                id=str(uuid.uuid4()),
                comment="This is a comment for the article.",  # Misalnya komentar default
                created_at=datetime.utcnow(),
                article_id=article_data['id'],
                user_id=random_user_id
            )
            db.session.add(comment)

        # Commit perubahan ke database
        db.session.commit()
        return ResponseBuilder.success("sukses",articles,status_code)