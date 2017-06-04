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
    parser.add_option("-o", "--output", dest="output", metavar="PATH",
                      help="path of hypothesis")
    (options, args) = parser.parse_args()   

    timer = Timer()
   
    # load data 
    print "loading data"
    movie_reviews = MovieReviews(
        'susan-granger'
    )
    timer.interval('Loaded Data')

    print "converting to arrays"
    X, Y = load_svmlight_file(movie_reviews.training_svm)
    movies = movie_reviews.all_movies
    X_predict, _ = load_svmlight_file(movie_reviews.predict_svm)
    timer.interval('converted to arrays')

    model_trainer = ModelTrainer(X, Y, movies, timer)
    model_trainer.calc_metrics()

    model_trainer.train_and_predict(X_predict)
    model_trainer.save(options.output)
