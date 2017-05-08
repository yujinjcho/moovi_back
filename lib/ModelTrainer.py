import json
import logistic_regression
import random_forest

class ModelTrainer(object):
    def __init__(self, X_train, X_test, movies, Y, timer):
        self.X_train = X_train
        self.X_test = X_test
        self.movies = movies
        self.Y = Y
        self.timer = timer
        self.predictions = {}

    def train_and_predict(self, model_type, tuning_parameters):
        active_model = self.select_model(model_type)
        active_model(self, tuning_parameters)

    def select_model(self, model_type):
        if model_type == 'logistic_regression':
            return logistic_regression.train_and_predict
        else:
            return random_forest.train_and_predict

    def store_predict(self, hypothesis, tuning_param):
        self.predictions[tuning_param] = dict(zip(self.movies, hypothesis))

    def save(self, output):
        with open(output, 'w') as f:
            f.write(json.dumps(self.predictions))
        



