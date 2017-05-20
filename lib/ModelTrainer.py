import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class ModelTrainer(object):
    def __init__(self, X_train, X_test, y_train, y_test, movies, timer):
        self.X_train = X_train
        self.X_test = X_test
        self.movies = movies
        self.y_train = y_train
	self.y_test = y_test
        self.timer = timer
        self.predictions = {}


    def train_and_predict(self, model_type, tuning_parameters):
        clf = LogisticRegression(C=0.01)
        clf.fit(self.X_train, self.y_train) 
        predicted = clf.predict(self.X_test)
        print 'accuracy: ', accuracy_score(self.y_test, predicted)


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
        



