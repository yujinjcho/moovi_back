import os
import json
import psycopg2
from psycopg2.extras import execute_batch

from lib.MovieReviews import MovieReviews
from lib.Timer import Timer
from lib.ModelTrainer import ModelTrainer
from recommender import MovieRecommender
from sklearn.datasets import load_svmlight_file

import config


def get_recommendations(data, timer):
    timer.interval('start')
    db = psycopg2.connect(**config.db_config)
    ratings = data['ratings']
    user = data['user_id']
    timer.interval('get request data')
    _write_to_db(ratings, user, db)
    timer.interval('write to db')
    recommendations = _calculate_recommendations(user)
    timer.interval('calculate recommendations')
    db.close()
    return json.dumps(recommendations)



def _calculate_recommendations(user):
    movie_reviews = MovieReviews(user)

    X, Y = load_svmlight_file(movie_reviews.training_svm)
    movies = movie_reviews.all_movies
    X_predict, _ = load_svmlight_file(movie_reviews.predict_svm)

    model_trainer = ModelTrainer(X, Y, movies, Timer())
    model_trainer.train_and_predict(X_predict)
    recommender = MovieRecommender(movie_reviews, model_trainer)
    return recommender.top_n(50)


def _write_to_db(ratings, user, db):
    new_query = 'INSERT INTO ratings (user_id, rotten_id, rating) VALUES ( %s, %s, %s)'

    rated_movies = _get_rated_movies(user, db)
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

def _get_rated_movies(user, db):
    query = "SELECT rotten_id from ratings WHERE user_id = %s"
    with db.cursor() as cur:
        cur.execute(query, (user,))
        results = cur.fetchall()
        return set([x[0] for x in results])





