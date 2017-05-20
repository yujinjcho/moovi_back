import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class ModelTrainer(object):
    def __init__(self, X_train, X_test, y_train, y_test, movies, timer):
        self.X_train = X_train
        self.X_test = X_test
        self.movies = movies
        self.y_train = y_train
	self.y_test = y_test
        self.timer = timer
        self.predictions = {}


    def train_and_predict(self):
        parameters = {'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0 ]} 

        print "Tuning params with GridSearch"
        clf = GridSearchCV(LogisticRegression(), parameters, n_jobs=-1, error_score=0) 
        clf.fit(self.X_train, self.y_train) 
        self.timer.interval("Finished tuning params")
       
        print "Predicting results"
        predicted = clf.predict(self.X_test)
        self.timer.interval("Finished Predictions\n")

	print 'Stats:'
	print '----------'
        print 'accuracy: ', accuracy_score(self.y_test.ravel(), predicted)
        print 'f1:', f1_score(self.y_test.ravel(), predicted)
        print 'precision: ', precision_score(self.y_test.ravel(), predicted)
        print 'recall: ', recall_score(self.y_test.ravel(), predicted)
	print '----------'



    def store_predict(self, hypothesis, tuning_param):
        self.predictions[tuning_param] = dict(zip(self.movies, hypothesis))

    def save(self, output):
        with open(output, 'w') as f:
            f.write(json.dumps(self.predictions))
        



