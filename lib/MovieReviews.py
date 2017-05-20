import json
from collections import defaultdict

class MovieReviews(object):
    def __init__(self, input_path, positive, negative, exclude=None):
        self.reviews = [json.loads(line) for line in self.load_data(input_path)]
        self.liked_movies = self.load_data(positive)
        self.disliked_movies = self.load_data(negative)
        self.seen_movies = self.liked_movies + self.disliked_movies
        self.critics = list(set([review['critic'] for review in self.reviews]))
        self.critic_ratings = self.critic_rating_mapping()

        if exclude:
            self.critic_ratings.pop(exclude, None)
            self.critics.remove(exclude)

        self.movie_mapping = self.make_movie_mapping()
        self.all_movies = self.movie_mapping.keys()
        self.create_predict_svm()
        self.create_svm()

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
        with open('training.svm', 'w') as f:
            for movie in self.seen_movies:
                if movie in self.liked_movies:
                    label = '1'
                else:
                    label = '-1'

                if movie in self.movie_mapping:
                    features = ' '.join(self.movie_mapping[movie])
                else:
                    features = ''
                f.write("{} {}\n".format(label, features))

    def create_predict_svm(self):
        with open('predict.svm', 'w') as f:
            for movie in self.movie_mapping:
                features = ' '.join(self.movie_mapping[movie])
                f.write("0 {} # {}\n".format(features, movie.encode('utf8')))


    def load_data(self, filename):
        with open(filename) as f:
            return [line.rstrip() for line in f]

    def critic_rating_mapping(self):
        ratings = {}

        for review in self.reviews:
            critic_name = review['critic']
            critic_ratings = ratings.get(critic_name, {})
            critic_ratings.update({ review['movie']: int(review['rating']) })
            ratings[critic_name] = critic_ratings

        return ratings
