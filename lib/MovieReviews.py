import os
import json
import StringIO
from collections import defaultdict
import psycopg2

class MovieReviews(object):
    def __init__(self, target_user):
        self.reviews = self.load_reviews()
        self.liked_movies = self.load_movies('1', target_user)
        self.disliked_movies = self.load_movies('-1', target_user)
        self.seen_movies = self.liked_movies + self.disliked_movies
        self.critics = list(set([review['critic'] for review in self.reviews]))
        self.critic_ratings = self.critic_rating_mapping()

        self.critic_ratings.pop(target_user, None)
        self.critics.remove(target_user)

        self.movie_mapping = self.make_movie_mapping()
        self.all_movies = self.movie_mapping.keys()
        self.predict_svm = self.create_predict_svm()
        self.training_svm = self.create_svm()


    def load_reviews(self):
        conn = psycopg2.connect(
           dbname=os.environ['DBNAME'],
           user=os.environ['PGUSER'],
           password=os.environ['PGPASSWORD'],
           port=os.environ['PGPORT'],
           host=os.environ['PGHOST']
        )
        

        cur = conn.cursor()  
        query = """select rotten_id, user_id, rating from ratings"""
        cur.execute(query)
        results = cur.fetchall()
        cur.close()
        conn.close()
        return [{
            'movie': result[0],
            'critic': result[1],
            'rating': result[2]
        } for result in results]


    def load_movies(self, classification, target_user):
        conn = psycopg2.connect(
            dbname=os.environ['DBNAME'],
            user=os.environ['PGUSER'],
            password=os.environ['PGPASSWORD'],
            port=os.environ['PGPORT'],
            host=os.environ['PGHOST']
        )
        
        cur = conn.cursor()  
        query = """select rotten_id from ratings
                 where user_id = %s and 
                 rating = %s"""
        query_data = (target_user, classification)
        cur.execute(query, query_data)
        results = cur.fetchall()
        cur.close()
        conn.close()
        return [result[0] for result in results]


    def make_movie_mapping(self):
        mapping = defaultdict(list)
        seen_movies = set(self.seen_movies)
        features = set()
        for i, critic in enumerate(self.critics):
            for movie, rating in self.critic_ratings[critic].iteritems():
                if movie in seen_movies:
                    features.add(critic)
        self.features = features

        for i, critic in enumerate(self.critics):
            for movie, rating in self.critic_ratings[critic].iteritems():
                if critic in self.features:
                    mapping[movie].append("{}:{}".format(i, rating))

        return dict(mapping)


    def create_svm(self):
        svm = []
        output = StringIO.StringIO()
        for movie in self.seen_movies:
            if movie in self.liked_movies:
                label = '1'
            else:
                label = '-1'

            if movie in self.movie_mapping:
                features = ' '.join(self.movie_mapping[movie])
            else:
                features = ''
            svm.append("{} {}".format(label, features))
        return StringIO.StringIO("\n".join(svm))


    def create_predict_svm(self):
        svm = []
        for movie in self.movie_mapping:
            features = ' '.join(self.movie_mapping[movie])
            svm.append("0 {} # {}".format(features, movie))
        return StringIO.StringIO("\n".join(svm))

    def critic_rating_mapping(self):
        ratings = {}

        for review in self.reviews:
            critic_name = review['critic']
            critic_ratings = ratings.get(critic_name, {})
            critic_ratings.update({ review['movie']: int(review['rating']) })
            ratings[critic_name] = critic_ratings

        return ratings
