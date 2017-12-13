import os
import json
from optparse import OptionParser
import psycopg2

class MovieRecommender(object):
    def __init__(self, movie_reviews, model_trainer):
        self.predictions = self.load_predictions(model_trainer)
        self.liked_movies = movie_reviews.liked_movies
        self.disliked_movies = movie_reviews.disliked_movies
        self.seen_movies = set(self.liked_movies + self.disliked_movies)
        self.movie_mapping = self.load_mapping()
        self.box_office = self.load_box_office()
    
    def load_predictions(self, model_trainer):
        return dict(
            zip(model_trainer.movies, model_trainer.predictions.tolist())
        )

    def load_mapping(self):
        conn = psycopg2.connect(
            dbname=os.environ['DBNAME'],
            user=os.environ['PGUSER'],
            password=os.environ['PGPASSWORD'],
            port=os.environ['PGPORT'],
            host=os.environ['PGHOST']
        )
        cur = conn.cursor()
        query = "select rotten_id, title from movies"
        cur.execute(query)
        results = cur.fetchall()
        cur.close()
        conn.close()
        return dict(results)

    def load_box_office(self):
        conn = psycopg2.connect(
            dbname=os.environ['DBNAME'],
            user=os.environ['PGUSER'],
            password=os.environ['PGPASSWORD'],
            port=os.environ['PGPORT'],
            host=os.environ['PGHOST']
        )

        query = """SELECT UNNEST(movies)
                   FROM (
                     SELECT *
                     FROM box_office
                     ORDER BY 
                        date_created DESC
                     LIMIT 1
                   ) i
        """
       
        with conn.cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
        return set(x[0] for x in results)

    def box_office_recommendations(self):
        predict_like = [
            (movie,scores) for movie, scores in self.predictions.items() 
            if scores[1] > scores[0] and movie not in self.seen_movies
            if movie in self.box_office
        ]
        predict_sorted = sorted(predict_like, key=lambda x: x[1][1], reverse=True)
        return [
            self.movie_mapping[movie[0]]
            for movie in predict_sorted
        ]


    def top_n(self, n):
        predict_like = [
            (movie,scores) for movie, scores in self.predictions.items() 
            if scores[1] > scores[0] and movie not in self.seen_movies
        ]
        predict_sorted = sorted(predict_like, key=lambda x: x[1][1], reverse=True)
        return self.recommendations(predict_sorted, n)

    def bottom_n(self, n):
        predict_dislike = [
            (movie,scores) for movie, scores in self.predictions.items() 
            if scores[0] > scores[1] and movie not in self.seen_movies
        ]
        predict_sorted = sorted(predict_dislike, key=lambda x: x[1][0], reverse=True)
        return self.recommendations(predict_sorted, n)


    def recommendations(self, predict_sorted, n):
        return [
            self.movie_mapping[predict_sorted[i][0]]
            for i in range(n)
        ]


    def rating_for_movie(self, movie):
        if movie in self.predictions:
            scores = self.predictions[movie]
            return {
                'like': scores[1],
                'dislike': scores[0]
            }

        return None
