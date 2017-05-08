import json
from optparse import OptionParser
from os import listdir
from collections import defaultdict
import matplotlib.pyplot as plt

def load_data(filename):
    with open(filename) as f:
        return [line.rstrip() for line in f] 

def init_stats():
    stats = {}
    chart_type = ['accuracy', 'precision', 'recall', 'f-score' ]
    for accuracy_type in chart_type:
        stats[accuracy_type] = defaultdict(list)
    return stats

def load_predictions(filename):
    with open(filename) as f:
        data = json.loads(f.read())
    predictions = {}
    for tuning_param in data:
        classified = dict([
            (movie, '1') 
            if rating['like'] > rating['dislike']
            else (movie, '2')
            for movie, rating in data[tuning_param].items()
        ])
        predictions[tuning_param] = classified
    return predictions

def evaluate_accuracy(predictions, positives, negatives, stats):
    total = len(positives) + len(negatives)
    for tuning_param in predictions:

        pos_correct = sum([ 1 if predictions[tuning_param][movie] == '1' else 0 for movie in positives])
        neg_correct = sum([ 1 if predictions[tuning_param][movie] == '2' else 0 for movie in negatives])
        correct = pos_correct + neg_correct

        pos_predicted = pos_correct + (len(negatives) - neg_correct)

        accuracy = correct / float(total) 
        precision = pos_correct / float(pos_predicted)
        recall = pos_correct / float(len(positives))
        f_score = 2 * precision * recall / (precision + recall)

        stats['accuracy'][tuning_param] = accuracy
        stats['precision'][tuning_param] = precision
        stats['recall'][tuning_param] = recall
        stats['f-score'][tuning_param] = f_score


        print "\nTuning Param Value {}".format(tuning_param)
        print "Model accurately predicted {} / {}".format(correct, total)
        print "Accuracy: {0:.1f}%".format(accuracy * 100)
        print "Precision: {0:.1f}%".format(precision * 100)
        print "Recall: {0:.1f}%".format(recall * 100)
        print "F-Score: {0:.2f}".format(f_score)

def chart_stats(stats):
    for stat_type, stat_values in stats.items():
        chart_data = sorted(stat_values.items(), key=lambda x:float(x[0]))
        x_array = [score[0] for score in chart_data]
        y_array = [score[1] for score in chart_data]
        plt.plot(x_array, y_array, label=stat_type, alpha=0.5)

    plt.xlabel('Tuning Param')
    plt.ylabel('%')
    plt.title('Prediction Metrics')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # parse options. inputs <-
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", metavar="PATH",
                      help="path to predicted classes")
    parser.add_option("-p", "--positive", dest="positive", metavar="PATH",
                      help="path to positive test set")
    parser.add_option("-n", "--negative", dest="negative", metavar="PATH",
                      help="path to negative test set")
    (options, args) = parser.parse_args()   
   
    positives = load_data(options.positive)
    negatives = load_data(options.negative)

    stats = init_stats()
    predictions = load_predictions(options.input)
    evaluate_accuracy(predictions, positives, negatives, stats)
    chart_stats(stats)
