from sklearn.ensemble import RandomForestClassifier

def train_and_predict(trainer, tuning_params):
    for param in tuning_params:
        print "Training for param value: {}".format(param)
        clf = RandomForestClassifier(
            n_jobs=2, 
            oob_score=True, 
            random_state=50, 
            max_features = "auto", 
            min_samples_leaf=param, 
            n_estimators=500
        )
        clf.fit(trainer.X_train, trainer.Y.ravel())
        trainer.timer.interval('trained model')

        print "Predicting"
        class_probabilities = clf.predict_proba(trainer.X_test)
        trainer.timer.interval('predicted')
        hypothesis = [
            {"like": predict[0], "dislike": predict[1]}
            for predict in class_probabilities
        ]
        trainer.store_predict(hypothesis, param)

