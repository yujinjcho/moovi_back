import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


class ModelTrainer(object):
    parameters = {'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0 ]} 

    def __init__(self, X, Y, movies, timer):
        self.X = X
        self.Y = Y
        self.movies = movies
        self.timer = timer

    def train_and_predict(self, X_predict):
        print "Tuning with all inputs"
        clf = GridSearchCV(LogisticRegression(), self.parameters, n_jobs=-1, error_score=0) 
        clf.fit(self.X, self.Y) 
        self.timer.interval("Finished tuning with all inputs")
        self.predictions = clf.predict_proba(X_predict)


    def calc_metrics(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=.25, random_state=1)

        print "Tuning params with GridSearch"
        clf = GridSearchCV(LogisticRegression(), self.parameters, n_jobs=-1, error_score=0) 
        clf.fit(X_train, y_train) 
        self.timer.interval("Finished tuning params")
       
        print "Predicting results"
        predicted = clf.predict(X_test)
        # predicted_scores = clf.predict_proba(self.X_test)
        self.timer.interval("Finished Predictions\n")

	print 'Stats:'
	print '----------'
        print 'accuracy: ', accuracy_score(y_test, predicted)
        print 'f1:', f1_score(y_test, predicted)
        print 'precision: ', precision_score(y_test, predicted)
        print 'recall: ', recall_score(y_test, predicted)
	print '----------'


    def save(self, output):
        with open(output, 'w') as f:
            movie_scores = dict(zip(self.movies, self.predictions.tolist()))
            f.write(json.dumps(movie_scores))
        



