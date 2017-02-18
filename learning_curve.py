""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

#data = load_digits()
#X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
#train_size=0.3)
#model = LogisticRegression(C=10**-10)
#model.fit(X_train, y_train)
#print("Train accuracy %f" %model.score(X_train,y_train))
#print("Test accuracy %f"%model.score(X_test,y_test))

def display_digits():
    digits = load_digits()
    fig = plt.figure()
    for i in range(10):
        subplot = fig.add_subplot(5, 2, i+1)
        subplot.matshow(numpy.reshape(digits.data[i], (8, 8)), cmap='gray')

    plt.show()


def train_model():
    # train models with training percentages between 5 and 90 (see
    # train_percentages) and evaluate the resultant accuracy for each.
    # You should repeat each training percentage num_trials times to smooth out
    # variability.
    # For consistency with the previous example use
    # model = LogisticRegression(C=10**-10) for your learner
    data = load_digits()
    num_trials = 10
    fig = plt.figure()
    train_percentages = range(5, 95, 5)
    #This is the axis for the graph
    test_accuracies = numpy.zeros(len(train_percentages))
    list_index_count = 0
    #for loop runs through all percentage values
    for train_set in train_percentages:
        list_index_count += 1
        accuracy_total = 0
        #for loop runs through 10 trials
        for i in range(num_trials):
            #print(i)
            #code runs through trianer and splits into train and test set
            #train size is the train set /100 so it will split it up based on
            #the percentage of the train_set to be used to train
            X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
            train_size= train_set/100)
            model = LogisticRegression(C=3**-10)
            model.fit(X_train, y_train)

            #calculates the accuracies for the test set.
            current_test_accuracy = model.score(X_test, y_test)
            #adds all the accuracies together so an average can be found.
            accuracy_total = accuracy_total + current_test_accuracy

        #this declares the list to be plotted.
        #It takes the sum of the accuracies and divides them by the trials
        test_accuracies[list_index_count-1] = accuracy_total/num_trials
        #list_of_averages.append([train_set, average_accuracy/num_trials])

    #graph the data given
    #plot with data of train_percentages and test_accuracies
    plt.plot(train_percentages, test_accuracies)
    plt.xlabel('Percentage of Data Used for Training')
    plt.ylabel('Accuracy on Test Set')
    plt.show()


if __name__ == "__main__":
    # Feel free to comment/uncomment as needed
    #display_digits()
    #train_model()
    train_model()
