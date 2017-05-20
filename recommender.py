import json
from optparse import OptionParser

class MovieRecommender(object):
    def __init__(self, hypothesis_path, positive, negative):
        self.predictions = self.load_hypothesis(hypothesis_path)
        self.liked_movies = self.load_annotations(positive)
        self.disliked_movies = self.load_annotations(negative)
        self.seen_movies = set(self.liked_movies + self.disliked_movies)
        
    def load_hypothesis(self, filename):
        with open(filename) as f:
            return json.loads(f.read())

    def load_annotations(self, filename):
        with open(filename) as f:
            return [line.rstrip() for line in f]

    def top_n(self, n):
        predict_like = [
            (movie,scores) for movie, scores in self.predictions.items() 
            if scores[1] > scores[0]
        ]
        predict_sorted = sorted(predict_like, key=lambda x: x[1][1], reverse=True)
        for i in range(n):
            print i+1, predict_sorted[i][0], predict_sorted[i][1][1]

    def bottom_n(self, n):
        predict_dislike = [
            (movie,scores) for movie, scores in self.predictions.items() 
            if scores[0] > scores[1]
        ]
        predict_sorted = sorted(predict_dislike, key=lambda x: x[1][0], reverse=True)
        for i in range(n):
            print i+1, predict_sorted[i][0], predict_sorted[i][1][0]
        
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
  
    recommender = MovieRecommender(options.hypothesis, options.positive, options.negative)
    recommender.top_n(20)
    recommender.bottom_n(20)
    print recommender.rating_for_movie('1198124-shutter_island')

