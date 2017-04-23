import json

class MovieReviews(object):
    def __init__(self, input_path, positive, negative):
        self.reviews = [json.loads(line) for line in self.load_data(input_path)]
        self.liked_movies = self.load_data(positive)
        self.disliked_movies = self.load_data(negative)
        self.seen_movies = self.liked_movies + self.disliked_movies
        self.critics = list(set([review['critic'] for review in self.reviews]))
        self.critic_ratings = self.critic_rating_mapping()

    def load_data(self, filename):
        with open(filename) as f:
            return [line.rstrip() for line in f]

    def critic_rating_mapping(self):
        ratings = {}

        for review in self.reviews:
            rating = { review['critic']: { review['movie']: int(review['rating']) } }
            ratings.update(rating)

        return ratings
