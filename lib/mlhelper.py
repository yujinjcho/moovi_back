import numpy as np

def create_inputs_matrix(seen_movies, critics, review_mapping):
    return np.array([
        create_inputs_row(movie, critics, review_mapping) for movie in seen_movies
    ])

def create_inputs_row(movie, critics, review_mapping):
    return [add_rating(critic, movie, review_mapping) for critic in critics]

def add_rating(critic, movie, review_mapping):
    if movie in review_mapping[critic]:
        return review_mapping[critic][movie]
    return 0

def create_target_matrix(seen_movies, positives, negatives):
    y = np.zeros((len(seen_movies), 1))
    positive_set = set(positives)
    negative_set = set(negatives)
    for i, movie in enumerate(seen_movies):
        if movie in positive_set:
            y[i] = 1
        elif movie in negative_set:
            y[i] = 2
        else:
            y[i] = 0
    return y

