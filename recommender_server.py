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
    
    def load_predictions(self, model_trainer):
        return dict(
            zip(model_trainer.movies, model_trainer.predictions.tolist())
        )

    def load_mapping(self):
        conn = psycopg2.connect('dbname=movie_rec')
        cur = conn.cursor()
        query = "select movie_id, title from movies"
        cur.execute(query)
        results = cur.fetchall()
        cur.close()
        conn.close()
        return dict([
            (result[0], result[1])
            for result in results
        ])


    def top_n(self, n):
        predict_like = [
            (movie,scores) for movie, scores in self.predictions.items() 
            if scores[1] > scores[0]
        ]
        predict_sorted = sorted(predict_like, key=lambda x: x[1][1], reverse=True)
        predict_sorted = [x for x in predict_sorted if x[0] not in self.seen_movies]
        return self.recommendations(predict_sorted, n)


    def bottom_n(self, n):
        predict_dislike = [
            (movie,scores) for movie, scores in self.predictions.items() 
            if scores[0] > scores[1]
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
