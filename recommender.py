import json
from optparse import OptionParser
import psycopg2

class MovieRecommender(object):
    def __init__(self, hypothesis_path, target_user):
        self.predictions = self.load_hypothesis(hypothesis_path)
        self.liked_movies = self.load_movies('1', target_user)
        self.disliked_movies = self.load_movies('-1', target_user)
        self.seen_movies = set(self.liked_movies + self.disliked_movies)
	self.movie_mapping = self.load_mapping()

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


    def load_movies(self, classification, target_user):
        conn = psycopg2.connect('dbname=movie_rec')
        cur = conn.cursor()
        query = """select movie_id from ratings
                 where user_id = %s and 
                 rating = %s"""
        query_data = (target_user, classification)
        cur.execute(query, query_data)
        results = cur.fetchall()
        cur.close()
        conn.close()
        return [result[0] for result in results]

    def load_hypothesis(self, filename):
        with open(filename) as f:
            return json.loads(f.read())

    def top_n(self, n):
        predict_like = [
            (movie,scores) for movie, scores in self.predictions.items() 
            if scores[1] > scores[0]
        ]
        predict_sorted = sorted(predict_like, key=lambda x: x[1][1], reverse=True)
        predict_sorted = [x for x in predict_sorted if x[0] not in self.seen_movies]
        self.print_recommendations(predict_sorted, n)


    def bottom_n(self, n):
        predict_dislike = [
            (movie,scores) for movie, scores in self.predictions.items() 
            if scores[0] > scores[1]
        ]
        predict_sorted = sorted(predict_dislike, key=lambda x: x[1][0], reverse=True)
        self.print_recommendations(predict_sorted, n)


    def print_recommendations(self, recommendations, n):
        for i in range(n):
            movie_id = recommendations[i][0]
            score = recommendations[i][1][1]
            title = self.movie_mapping[int(movie_id)]
            print i+1, title, score 
       

    def rating_for_movie(self, movie):
        if movie in self.predictions:
            scores = self.predictions[movie]
            return {
                'like': scores[1],
                'dislike': scores[0]
            }

        return None

if __name__ == '__main__':
    # parse options. inputs <-
    parser = OptionParser()
    parser.add_option("-y", "--hypothesis", dest="hypothesis", metavar="PATH",
                      help="path to predicted values")
    parser.add_option("-p", "--positive", dest="positive", metavar="PATH",
                      help="path to positive test set")
    parser.add_option("-n", "--negative", dest="negative", metavar="PATH",
                      help="path to negative test set")
    (options, args) = parser.parse_args()   
  
    recommender = MovieRecommender(options.hypothesis, 'susan-granger')
    recommender.top_n(100)
    recommender.bottom_n(20)

