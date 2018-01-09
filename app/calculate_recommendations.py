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

parameters = {'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0 ]} 

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
    return old_calc(user)

def old_calc(user):
    movie_reviews = MovieReviews(user)
    X, Y = load_svmlight_file(movie_reviews.training_svm)
    movies = movie_reviews.all_movies
    X_predict, _ = load_svmlight_file(movie_reviews.predict_svm)

    model_trainer = ModelTrainer(X, Y, movies, Timer())
    model_trainer.train_and_predict(X_predict)
    recommender = MovieRecommender(movie_reviews, model_trainer)
    return recommender.box_office_recommendations()

def new_calc(user):
    # make db request to get each movie in svm format

    # select 
    #   t.movie_id,
    #   t.svm_text,
    #   r.label
    # from (
    #   select 
    #     movie_id, 
    #     array_to_string(array_agg(user_id || ':' || label), ' ') as svm_text 
    #   from (
    #     select * 
    #     from rating_test 
    #     where user_id != 1 
    #     order by user_id
    #   ) as q 
    #   group by movie_id
    # ) as t left join rating_test r on t.movie_id = r.movie_id and r.user_id = 1;
 
    # movie_svm will be an array of tuples (true_label, svm_text, movie_id)
    # movie_svm = [
    #     (1, '0:1 2:-1', 'movie1_id'),
    #     (None, '0:1 2:-1', 'movie2_id')
    # ]

    # svm = split_svm(movie_svm)
    # actually only need to predict box_office movies


    train_svm = create_train_svm()
    predict_svm, movies = create_predict_svm()

    X, Y = load_svmlight_file(svm['train_svm'])
    X_predict, _ = load_svmlight_file(svm['predict_svm'])
    classifier = train_model(X, y)
    predictions = classifier.predict_proba(X_predict)
    recommendations = format_recommendations(predictions, svm['predict_movies'])

def split_svm(movie_svm):
    train_svm = []
    train_movies = []
    predict_svm = []
    predict_movies = []
    for svm in movie_svm:
        if svm[0]:
            train_svm.append("{} {}".format(svm[0], svm[1]))
            train_movies.append(svm[2])
        else:
            predict_svm.append("{} {}".format(0, svm[1]))
            predict_movies.append(svm[2])
    return {
        'train_svm': train_svm,
        'train_movies': train_movies,
        'predict_svm': predict_svm,
        'predict_movies': predict_movies
    }

def train_model(x y):
    clf = GridSearchCV(LogisticRegression(random_state=1), self.parameters, n_jobs=-1, error_score=0)
    clf.fit(x, y)
    return clf

def format_recommendations(predictions, movies):
    box_office_movies = db_manager.load_box_office()
    # movie_titles = db_manager.load_movie_titles()
    
    # Instead of zip / can just iterate and yield if score[1] > score[0]
    # AND in box_office

    movie_predictions = dict(zip(movies, predictions.tolist()))
    predict_like = [
        (movie, scores) for movie, scores in movie_predictions.items() 
        if scores[1] > scores[0] 
        if movie in bos_office_movies
    ]
    predict_sorted = sorted(predict_like, key=lambda x: x[1][1], reverse=True)
    return predicted_sorted


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
