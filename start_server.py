import json
from flask import Flask, request, g
import psycopg2

from lib.MovieReviews import MovieReviews
from lib.Timer import Timer
from lib.ModelTrainer import ModelTrainer
from recommender import MovieRecommender
from sklearn.datasets import load_svmlight_file

app = Flask(__name__)


@app.route('/api', methods=['post'])
def api():
    ratings = json.loads(request.form.get('ratings', type=str))
    user = request.form.get('device_id', type=str)
    _write_to_db(ratings, user)
    recommendations = _calculate_recommendations(user)
    return json.dumps(recommendations)


def _calculate_recommendations(user):
    movie_reviews = MovieReviews(user)
    X, Y = load_svmlight_file(movie_reviews.training_svm)
    movies = movie_reviews.all_movies
    X_predict, _ = load_svmlight_file(movie_reviews.predict_svm)

    model_trainer = ModelTrainer(X, Y, movies, Timer())
    model_trainer.train_and_predict(X_predict)
    recommender = MovieRecommender(movie_reviews, model_trainer)
    return {'recommendations': recommender.top_n(50)}


def _write_to_db(ratings, user):
    query = """INSERT INTO ratings (user_id, movie_id, rating)
               SELECT %s, %s, %s
                WHERE NOT EXISTS ( SELECT rating_id FROM ratings 
                                    WHERE user_id = %s AND
                                          movie_id = %s)
    """

    db = get_db()
    with db.cursor() as cur:
        for rating in ratings:
            upload_data = (
                user, 
                rating['movie_id'], 
                rating['rating'], 
                user, 
                rating['movie_id']
             )
            cur.execute(query, upload_data)
    db.commit()


def get_db():
    if not hasattr(g, 'db'):
        g.db = psycopg2.connect('dbname=movie_rec')
    return g.db


@app.teardown_appcontext
def close_db(error):
    if hasattr(g, 'db'):
        g.db.close()


if __name__ ==  '__main__':
    app.run(debug=True)
