import json
import numpy as np
from optparse import OptionParser

from lib.MovieReviews import MovieReviews
from lib.ModelTrainer import ModelTrainer
from lib.mlhelper import create_inputs_matrix, create_test_inputs_matrix, create_target_matrix
from lib.Timer import Timer

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split


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

    print "converting to arrays"
    X, Y = load_svmlight_file('test_output')
    movies = movie_reviews.all_movies
    X_predict, _ = load_svmlight_file('test_output_2')
    timer.interval('converted to arrays')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.25, random_state=1)

    model_trainer = ModelTrainer(X_train, X_test, y_train, y_test, movies, timer)
    lmbdas = [0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    model_trainer.train_and_predict('logistic_regression', lmbdas)
