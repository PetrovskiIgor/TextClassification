from sklearn import datasets
from sklearn import svm
from time import time




digits = datasets.load_digits()

numSamples = len(digits.data)


training_factor = 0.7

lastIndexTraining = int(training_factor * numSamples)

svmClassifier = svm.SVC(gamma=0.001,C=100.)



begin = time()

print svmClassifier.fit(digits.data[:lastIndexTraining], digits.target[:lastIndexTraining])


end = time()

print 'Learning ended in %f seconds.' % (end - begin)

our_predictions = svmClassifier.predict(digits.data[lastIndexTraining:])


total = len(our_predictions)

numCorrect = 0
for i in range(0, total):
    if our_predictions[i] == digits.target[lastIndexTraining + i]:
        numCorrect += 1



print 'percentage correct: %f\n' % (100 * (numCorrect * 1.0)/total)




