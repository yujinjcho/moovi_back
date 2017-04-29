import json
from optparse import OptionParser

def load_data(filename):
    with open(filename) as f:
        return [line.rstrip() for line in f] 

def load_predictions(filename):
    with open(filename) as f:
        data = json.loads(f.read())
    classified = dict([
	(movie, '1') 
	if rating['like'] > rating['dislike']
	else (movie, '2')
	for movie, rating in data.items()
    ])
    return classified

def evaluate_accuracy(predictions, positives, negatives):
    correct  = 0
    total = len(positives) + len(negatives)

    pos_correct = sum([ 1 if predictions[movie] == '1' else 0 for movie in positives])
    neg_correct = sum([ 1 if predictions[movie] == '2' else 0 for movie in negatives])
    correct = pos_correct + neg_correct

    print "\nModel accurately predicted {} / {}".format(correct, total)
    print "Accuracy: {0:.1f}%\n".format((correct / float(total)) * 100)


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
   
    predictions = load_predictions(options.input)
    positives = load_data(options.positive)
    negatives = load_data(options.negative)
    evaluate_accuracy(predictions, positives, negatives) 

