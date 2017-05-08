import json

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
