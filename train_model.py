import json
import numpy as np
from optparse import OptionParser

from lib.moviereviews import MovieReviews
from lib.mlhelper import create_inputs_matrix, create_test_inputs_matrix, create_target_matrix
from lib.multi_class import one_vs_all, predict_all
from lib.Timer import Timer

def load_data(filename):
    with open(filename) as f:
        return [line.rstrip() for line in f]


def save_data(data, output):
    with open(output, 'w') as f:
        f.write(json.dumps(dict(data)))


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

    timer = Timer()
   
    # load data 
    print "loading data"
    movie_reviews = MovieReviews(
        options.inputs,
        options.positive,
        options.negative
    )
    timer.interval('Loaded Data')

    print "creating inputs matrix"
    X = create_inputs_matrix(
        movie_reviews.seen_movies,
        movie_reviews.critics,
        movie_reviews.critic_ratings
    )
    timer.interval('Loaded X-matrix')

    print "creating target matrix"
    Y = create_target_matrix(
        movie_reviews.seen_movies,
        movie_reviews.liked_movies
    )
    timer.interval('Loaded y-matrix')

    print "creating test inputs matrix"
    movies, X_test = create_test_inputs_matrix(
        movie_reviews.critics,
        movie_reviews.critic_ratings
    )
    timer.interval('Created test inputs matrix')

    # output <- params, ML formatted data
    learning_rate = 1
    num_classifications = 2

    print "calculating theta"
    all_theta = one_vs_all(X, Y, num_classifications, learning_rate)
    timer.interval('Calculated theta')

    print "predicting"
    hypothesis = predict_all(X_test, all_theta)
    timer.interval('Made prediction')

    print "saving hypothesis"
    save_data(zip(movies, hypothesis), options.output)
    timer.interval('Finished Saving')

