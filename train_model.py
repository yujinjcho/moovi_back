import json
import numpy as np
from optparse import OptionParser

from lib.MovieReviews import MovieReviews
from lib.ModelTrainer import ModelTrainer
from lib.Timer import Timer

from sklearn.datasets import load_svmlight_file


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
    X, Y = load_svmlight_file('training.svm')
    movies = movie_reviews.all_movies
    X_predict, _ = load_svmlight_file('predict.svm')
    timer.interval('converted to arrays')

    model_trainer = ModelTrainer(X, Y, movies, timer)
    model_trainer.calc_metrics()

    model_trainer.train_and_predict(X_predict)
    model_trainer.save('predictions.json')


