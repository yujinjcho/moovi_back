import json
import numpy as np
from optparse import OptionParser

from lib.MovieReviews import MovieReviews
from lib.ModelTrainer import ModelTrainer
from lib.mlhelper import create_inputs_matrix, create_test_inputs_matrix, create_target_matrix
import lib.logistic_regression
import lib.random_forest
from lib.Timer import Timer


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
        options.negative,
        'susan-granger'
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

    model_trainer = ModelTrainer(X, X_test, movies, Y, timer)
    # lmbdas = [0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    # model_trainer.train_and_predict('logistic_regression', lmbdas)

    min_leaves = [1, 5, 10, 20, 50, 100, 200]
    model_trainer.train_and_predict('random_forest', min_leaves)
    model_trainer.save(options.output)
