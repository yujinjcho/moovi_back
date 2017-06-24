import os
import json
from flask import Flask, request, g
import psycopg2
from psycopg2.extras import execute_batch, execute_values

from lib.MovieReviews import MovieReviews
from lib.Timer import Timer
from lib.ModelTrainer import ModelTrainer
from recommender import MovieRecommender
from sklearn.datasets import load_svmlight_file

app = Flask(__name__)

@app.route('/test', methods=['get', 'post'])
def test():
    return data['user_id'] + json.dumps(data['ratings'])

@app.route('/api/start', methods=['get'])
def start():
    return _top_movies()


@app.route('/api/refresh', methods=['post'])
def refresh():
    timer = Timer()
    data = request.get_json()
    timer.interval('get request data')
    ratings = data['ratings']
    user = data['user_id']
    _write_to_db(ratings, user)
    timer.interval('write to db')
    recommendations = _calculate_recommendations(user)
    timer.interval('calculate recommendations')
    return json.dumps(recommendations)


def _top_movies(user=None):
    db = get_db()
    if not user:
        query = """ SELECT m.title, m.rotten_id, m.image_url, count(r.rating_id)
                      FROM movies m, ratings r
                     WHERE m.rotten_id = r.rotten_id
                  GROUP BY m.title, m.rotten_id, m.image_url
                  ORDER BY COUNT(r.rating_id) DESC
                  LIMIT 2000
        """
        top_movies_to_rate = []
        with db.cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
            for result in results:
                movie = {
                    'title': result[0],
                    'movieId': result[1],
                    'photoUrl': result[2]
                }
                top_movies_to_rate.append(movie)

        return json.dumps(top_movies_to_rate)


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
    # need to filter out ones already in there
    new_query = 'INSERT INTO ratings (user_id, rotten_id, rating) VALUES ( %s, %s, %s)'

    db = get_db()
    upload_data = [
        (
            user, 
            x['movie_id'], 
            x['rating'], 
        )
        for x in ratings
    ]
    with db.cursor() as cur:
        execute_batch(cur, new_query, upload_data)
    db.commit()


def get_db():
    if not hasattr(g, 'db'):
        g.db = psycopg2.connect(
            dbname=os.environ['DBNAME'],
            user=os.environ['PGUSER'],
            password=os.environ['PGPASSWORD'],
            port=os.environ['PGPORT'],
            host=os.environ['PGHOST']
        )
    return g.db


@app.teardown_appcontext
def close_db(error):
    if hasattr(g, 'db'):
        g.db.close()


if __name__ ==  '__main__':
    app.run(debug=True)
