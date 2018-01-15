import os
import json
from StringIO import StringIO
from collections import defaultdict
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV
import psycopg2

from sklearn.datasets import load_svmlight_file

class MovieReviews(object):
    def __init__(self, target_user):

        self.conn = psycopg2.connect(
           dbname=os.environ['DBNAME'],
           user=os.environ['PGUSER'],
           password=os.environ['PGPASSWORD'],
           port=os.environ['PGPORT'],
           host=os.environ['PGHOST']
        )

        self.user_num = target_user
        

        # self.reviews = self.load_reviews()
        # self.liked_movies = self.load_movies('1', target_user)
        # self.disliked_movies = self.load_movies('-1', target_user)

        # self.seen_movies = self.liked_movies + self.disliked_movies
        # self.critics = list(set([review['critic'] for review in self.reviews]))
        # self.critic_ratings = self.critic_rating_mapping()

        # self.critic_ratings.pop(target_user, None)
        # self.critics.remove(target_user)

        # self.movie_mapping = self.make_movie_mapping()




        # self.reviews = self.load_reviews()
        # self.all_movies = self.movie_mapping.keys()
        # self.predict_svm = self.create_predict_svm()
        # self.training_svm = self.create_svm()

    def test_svm_setup(self):
        with self.conn.cursor() as cur:
            max_movie_num, max_critic_num = self.matrix_dimensions(cur)
            reviews = self.load_review(cur)
            matrix, user_ratings = self.generate_matrix(reviews, self.user_num, max_movie_num, max_critic_num)
            train_svm, predict_svm, movies = self.create_svm(matrix, user_ratings)
            X_train, y_train = load_svmlight_file(train_svm, n_features=max_critic_num)
            X_predict, _ = load_svmlight_file(predict_svm)
            predictions = self.train_model(X_train, y_train, X_predict)

        return X_train, y_train, X_predict, movies, predictions

    def matrix_dimensions(self, cur):
        query = "select max(movie_num) from movies_test"
        cur.execute(query)
        movie_max_num = cur.fetchone()[0]
    
        query = "select max(user_num) from users"
        cur.execute(query)
        critic_max_num = cur.fetchone()[0]
    
        return (movie_max_num, critic_max_num)

    def load_review(self, cur):
        query = """
            select movie_num, user_num, rating 
            from ratings_test where rating != '0' 
            order by user_num, movie_num
        """
        cur.execute(query)
        result = cur.fetchall()
        return result


    def generate_matrix(self, reviews, user_num, max_movie_num, max_critic_num):
        matrix = [[] for i in range(max_movie_num)]
        user_ratings = {}
        previous_movie_num = None
        previous_critic_num = None
    
        for review in reviews:
            movie_num = review[0]
            critic_num = review[1]
            label = review[2]
    
            if critic_num == user_num:
                user_ratings[movie_num] = label
            elif not (movie_num == previous_movie_num and critic_num == previous_critic_num):
                matrix[movie_num - 1].append("{}:{}".format(critic_num, label))
    
            previous_movie_num = movie_num
            previous_critic_num = critic_num
    
        return matrix, user_ratings


    def create_svm(self, matrix, user_ratings):
        train_svm = []
        predict_svm = []
        movies = []
    
        for movie_index, movie in enumerate(matrix):
    
            movie_num = movie_index + 1
            if len(movie) > 0:
                svm_text = ' '.join(movie)
                if movie_num in user_ratings:
                    label = user_ratings.get(movie_num, "0")
                    train_svm.append(label + ' ' + svm_text )
    
                predict_svm.append("0 " + svm_text)
                movies.append(movie_num)
    
        return StringIO('\n'.join(train_svm)), StringIO('\n'.join(predict_svm)), movies


    def train_model(self, X, y, X_predict):
        parameters = {'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0 ]}
        clf = GridSearchCV(LogisticRegression(random_state=1), parameters, n_jobs=-1, error_score=0) 
        clf.fit(X, y)
        return clf.predict_proba(X_predict)







#     def load_reviews(self):
#         conn = psycopg2.connect(
#            dbname=os.environ['DBNAME'],
#            user=os.environ['PGUSER'],
#            password=os.environ['PGPASSWORD'],
#            port=os.environ['PGPORT'],
#            host=os.environ['PGHOST']
#         )
#         
# 
#         cur = conn.cursor()  
#         query = """select rotten_id, user_id, rating from ratings where rating != '0'"""
#         cur.execute(query)
#         results = cur.fetchall()
#         cur.close()
#         conn.close()
#         return [{
#             'movie': result[0],
#             'critic': result[1],
#             'rating': result[2]
#         } for result in results]
# 

    # def load_movies(self, classification, target_user):
    #     conn = psycopg2.connect(
    #         dbname=os.environ['DBNAME'],
    #         user=os.environ['PGUSER'],
    #         password=os.environ['PGPASSWORD'],
    #         port=os.environ['PGPORT'],
    #         host=os.environ['PGHOST']
    #     )
    #     
    #     cur = conn.cursor()  
    #     query = """select rotten_id from ratings
    #              where user_id = %s and 
    #              rating = %s"""
    #     query_data = (target_user, classification)
    #     cur.execute(query, query_data)
    #     results = cur.fetchall()
    #     cur.close()
    #     conn.close()
    #     return [result[0] for result in results]



    # def make_movie_mapping(self):
    #     """ Creates mapping e.g.
    #         e.g. {movie_id: [feature1:1, feature2:0, ...]}

    #     """
    #     mapping = defaultdict(list)
    #     seen_movies = set(self.seen_movies)
    #     features = set()
    #     for i, critic in enumerate(self.critics):
    #         for movie, rating in self.critic_ratings[critic].iteritems():
    #             if movie in seen_movies:
    #                 features.add(critic)
    #     self.features = features

    #     for i, critic in enumerate(self.critics):
    #         for movie, rating in self.critic_ratings[critic].iteritems():
    #             if critic in self.features:
    #                 mapping[movie].append("{}:{}".format(i, rating))

    #     return dict(mapping)


    # def create_svm(self):
    #     svm = []
    #     output = StringIO.StringIO()
    #     for movie in self.seen_movies:
    #         if movie in self.liked_movies:
    #             label = '1'
    #         else:
    #             label = '-1'

    #         if movie in self.movie_mapping:
    #             features = ' '.join(self.movie_mapping[movie])
    #         else:
    #             features = ''
    #         svm.append("{} {}".format(label, features))
    #     return StringIO.StringIO("\n".join(svm))


    # def create_predict_svm(self):
    #     svm = []
    #     for movie in self.movie_mapping:
    #         features = ' '.join(self.movie_mapping[movie])
    #         svm.append("0 {} # {}".format(features, movie))
    #     return StringIO.StringIO("\n".join(svm))

    # def critic_rating_mapping(self):
    #     """ Creates mapping of critics and their movie ratings
    #         e.g. {critic_id1: {movie_id1: rating1, movie_id2: rating2}}
    #     
    #     """

    #     ratings = {}

    #     for review in self.reviews:
    #         critic_name = review['critic']
    #         critic_ratings = ratings.get(critic_name, {})
    #         critic_ratings.update({ review['movie']: int(review['rating']) })
    #         ratings[critic_name] = critic_ratings

    #     return ratings
