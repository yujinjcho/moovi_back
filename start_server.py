import os
import json
from flask import Flask, request, g
import psycopg2
from psycopg2.extras import execute_batch

from lib.MovieReviews import MovieReviews
from lib.Timer import Timer
from lib.ModelTrainer import ModelTrainer
from recommender import MovieRecommender
from sklearn.datasets import load_svmlight_file

import config

app = Flask(__name__)

timer = Timer()

@app.route('/test', methods=['get', 'post'])
def test():
    return data['user_id'] + json.dumps(data['ratings'])


@app.route('/api/start/', methods=['get'], defaults={'user_id':None})
@app.route('/api/start/<user_id>', methods=['get']) # think might not need this
def start(user_id):
    return _top_movies(user_id)


@app.route('/api/recommendations', methods=['post'])
def recommendations_api():
    timer.interval('start of refresh')
    data = request.get_json()
    timer.interval('get request data')
    ratings = data['ratings']
    user = data['user_id']
    _write_to_db(ratings, user)
    timer.interval('write to db')
    recommendations = _calculate_recommendations(user)
    timer.interval('calculate recommendations')
    return json.dumps(recommendations)


@app.route('/api/refresh', methods=['post'])
def refresh():
    data = request.get_json()
    ratings = data['ratings']
    user = data['user_id']
    not_rated_movies = data['not_rated_movies']
    _write_to_db(ratings, user)
    return _top_movies(user, not_rated_movies)


def _top_movies(user=None, not_rated_movies=[]):
    db = get_db()
    if not user:
        query = """ SELECT m.title, m.rotten_id, m.image_url, count(r.rating_id)
                      FROM movies m, ratings r
                     WHERE m.rotten_id = r.rotten_id
                  GROUP BY m.title, m.rotten_id, m.image_url
                  ORDER BY COUNT(r.rating_id) DESC
                  LIMIT 50
        """
        with db.cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
        N = 50
    else:
        query = """ SELECT m.title, m.rotten_id, m.image_url, count(r.rating_id)
                      FROM movies m, ratings r
                     WHERE m.rotten_id = r.rotten_id
                       AND m.rotten_id NOT IN (SELECT rotten_id FROM ratings WHERE user_id = %s)
                  GROUP BY m.title, m.rotten_id, m.image_url
                  ORDER BY COUNT(r.rating_id) DESC
                  LIMIT 100
        """
        with db.cursor() as cur:
            cur.execute(query, (user,))
            results = cur.fetchall()
        N = 25

    movies_to_exclude = set(not_rated_movies)
    top_movies_to_rate = []
    for result in results:
        if result[1] not in movies_to_exclude:
            movie = {
                'title': result[0],
                'movieId': result[1],
                'photoUrl': result[2]
            }
            top_movies_to_rate.append(movie)

    return json.dumps(top_movies_to_rate[:N])


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
    new_query = 'INSERT INTO ratings (user_id, rotten_id, rating) VALUES ( %s, %s, %s)'

    db = get_db()
    rated_movies = _get_rated_movies(user)
    upload_data = [
        (
            user, 
            x['movie_id'], 
            x['rating'], 
        )
        for x in ratings
        if x['movie_id'] not in rated_movies
    ]
    with db.cursor() as cur:
        execute_batch(cur, new_query, upload_data)
    db.commit()

def _get_rated_movies(user):
    query = "SELECT rotten_id from ratings WHERE user_id = %s"
    db = get_db()
    with db.cursor() as cur:
        cur.execute(query, (user,))
        timer.interval('query rated movies')
        results = cur.fetchall()
        return set([x[0] for x in results])

def get_db():
    if not hasattr(g, 'db'):
        g.db = psycopg2.connect(**config.db_config)
    return g.db


@app.teardown_appcontext
def close_db(error):
    if hasattr(g, 'db'):
        g.db.close()


if __name__ ==  '__main__':
    app.run(debug=True)
