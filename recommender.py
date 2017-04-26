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
        like_predict = [
            pred for pred in self.predictions.items() 
            if pred[1]['like'] > pred[1]['dislike']
        ]
        print "\nTop {} Movie Recommendations".format(n)
        self.sorted_n(like_predict, n, 'like')

    def bottom_n(self, n):
        dislike_predict = [
            pred for pred in self.predictions.items() 
            if pred[1]['dislike'] > pred[1]['like']
        ]
        print "\nBottom {} Movie Recommendations".format(n)
        self.sorted_n(dislike_predict, n, 'dislike')
        

    def sorted_n(self, preds, n, rating):
        sorted_predictions = sorted(
            preds, 
            key=lambda x: x[1][rating], 
            reverse=True
        )
        not_seen = [pred for pred in sorted_predictions if pred[0] not in self.seen_movies]
        for i in range(min(n, len(not_seen))):
            print i+1, not_seen[i][0]

    def rating_for_movie(self, movie):
        if movie in self.predictions:
            return self.predictions[movie]
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
    recommender.top_n(200)
    recommender.bottom_n(200)
    print recommender.rating_for_movie('1198124-shutter_island')

