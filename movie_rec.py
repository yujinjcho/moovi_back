import json
import numpy as np
from optparse import OptionParser

from lib.moviereviews import MovieReviews
from lib.mlhelper import create_inputs_matrix, create_test_inputs_matrix, create_target_matrix
from lib.multi_class import one_vs_all, predict_all


def load_data(filename):
    with open(filename) as f:
        return [line.rstrip() for line in f]


def save_data(data, output):
    with open(output, 'w') as f:
        f.write("\n".join(["{}\t{}".format(*item) for item in data]))


if __name__ == '__main__':
    # parse options. inputs <-
    parser = OptionParser()
    parser.add_option("-i", "--inputs", dest="inputs", metavar="PATH",
                      help="inputs for X")
    parser.add_option("-p", "--positive", dest="positive", metavar="PATH",
                      help="examples of positive target values")
    parser.add_option("-n", "--negative", dest="negative", metavar="PATH",
                      help="examples of negative target values")
    parser.add_option("-o", "--output", dest="output", metavar="PATH",
                      help="path of hypothesis")
    (options, args) = parser.parse_args()   
   
    # load data 
    print "loading data"
    movie_reviews = MovieReviews(
        options.inputs,
        options.positive,
        options.negative
    )

    # ML formatted data <- data 
    print "creating inputs matrix"
    X = create_inputs_matrix(
        movie_reviews.seen_movies,
        movie_reviews.critics,
        movie_reviews.critic_ratings
    )

    print "creating target matrix"
    Y = create_target_matrix(
        movie_reviews.seen_movies,
        movie_reviews.liked_movies
    )

    print "creating test inputs matrix"
    movies, X_test = create_test_inputs_matrix(
        movie_reviews.critics,
        movie_reviews.critic_ratings
    )

    # output <- params, ML formatted data
    learning_rate = 1
    num_classifications = 2

    print "calculating theta"
    all_theta = one_vs_all(X, Y, num_classifications, learning_rate)

    print "predicting"
    hypothesis = [x[0] for x in predict_all(X_test, all_theta).tolist()]

    print "saving hypothesis"
    save_data(zip(movies, hypothesis), options.output)

