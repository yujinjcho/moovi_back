import os
import json
from collections import defaultdict
from optparse import OptionParser
import psycopg2

class MovieRecommender(object):
    def __init__(self, movie_reviews, model_trainer):
        self.predictions = self.load_predictions(model_trainer)
        # self.liked_movies = movie_reviews.liked_movies
        # self.disliked_movies = movie_reviews.disliked_movies
        # self.seen_movies = set(self.liked_movies + self.disliked_movies)
        self.movie_mapping = self.load_mapping()
        self.target_movies = self.load_target_movies()
	self.recommendation_limit = 50
    
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

    def load_box_office(self, cur):
        query = """SELECT UNNEST(movies)
                   FROM (
                     SELECT *
                     FROM box_office
                     ORDER BY 
                        date_created DESC
                     LIMIT 1
                   ) i
        """
       
        cur.execute(query)
        results = cur.fetchall()
        return set(x[0] for x in results)


    def load_target_movies(self):
        conn = psycopg2.connect(
            dbname=os.environ['DBNAME'],
            user=os.environ['PGUSER'],
            password=os.environ['PGPASSWORD'],
            port=os.environ['PGPORT'],
            host=os.environ['PGHOST']
        )

        query = """
        
            SELECT streamer, rotten_id
            FROM streaming_movies
        
        """
       
        with conn.cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
	    box_office = self.load_box_office(cur)
	
	movies_by_provider = defaultdict(set)
	for x in results:
	    movies_by_provider[x[0]].add(x[1])
	
	movies_by_provider['box_office'] = box_office 
	return movies_by_provider

	
    def target_recommendations(self):

        predict_like = [(movie,scores) for movie, scores in self.predictions.items()]
        predict_sorted = sorted(predict_like, key=lambda x: x[1][1], reverse=True)

	recommendations = defaultdict(list)
	for movie_prediction in predict_sorted:
	    movie_id = movie_prediction[0]
	    score = movie_prediction[1][1]

	    for streamer, movies in self.target_movies.iteritems():
		if len(recommendations[streamer]) < self.recommendation_limit and movie_id in movies:
	            recommendations[streamer].append((self.movie_mapping[movie_id], score))
	
	return dict(recommendations)
