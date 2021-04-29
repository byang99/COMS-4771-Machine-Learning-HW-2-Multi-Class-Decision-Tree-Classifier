import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTree


def read_data(full_file_path):
    mat = scipy.io.loadmat(full_file_path)

    # Variable X is the 10,000 x 784 data matrix, each row
    # is a sample image of a hand-drawn digit
    # Variable Y is the 10,000 x 1 label vector where the ith
    # entry indicates the label of the ith sample image in X

    data = mat['X']
    label_vector = mat['Y']

    return data, label_vector


def split_data(data, label_vector, train_ratio):
    data_train, data_test, label_train, label_test = train_test_split(data, label_vector,
                                                                      train_size=train_ratio, random_state=42)

    train_matrix = np.concatenate((data_train, label_train), axis=1)
    test_matrix = np.concatenate((data_test, label_test), axis=1)

    return train_matrix, test_matrix


def test(classifier, test_matrix, k):
    labels = test_matrix[:, -1]
    num_correct = 0
    total = len(test_matrix)

    # for every image
    for row in range(len(test_matrix)):
        # get image vector
        image_vector = test_matrix[row][0:-1]

        # classify image
        label_prediction = classifier.classify(image_vector, k)

        # check if prediction is correct
        if label_prediction == labels[row]:
            num_correct += 1

    # compute and return accuracy
    accuracy = num_correct / total
    return accuracy


def __main__():
    full_file_path = "C:/Users/Brian/Desktop/Columbia University/Spring 2021/" \
                      "Machine Learning COMS 4771/HW 2 COMS 4771/digits.mat"

    # full_file_path = "./digits.mat"
    # read in image data
    data, label_vector = read_data(full_file_path)

    split_train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    max_depth = 50
    K = [i for i in range(1, max_depth + 1)]
    for train_ratio in split_train_ratios:
        # split data
        train_matrix, test_matrix = split_data(data, label_vector, train_ratio)
        print()
        print("********************************************************************")
        print("*                        "
              "Train/Test: ", math.ceil(train_ratio * 100), "/",
              100 - math.ceil(train_ratio * 100), "                     *")
        print("********************************************************************")

        # create decision tree
        dt = DecisionTree(train_matrix, max_depth=np.max(K))

        # get results for each K value
        train_error = []
        test_error = []
        for k in K:
            train_accuracy = test(dt, train_matrix, k)
            print("Train accuracy for depth ", k, " = ", train_accuracy)
            train_error.append(1 - train_accuracy)

        print()

        for k in K:
            test_accuracy = test(dt, test_matrix, k)
            print("Test accuracy for depth ", k, " = ", test_accuracy)
            test_error.append(1 - test_accuracy)

        fig1 = plt.figure()
        ax1 = fig1.add_axes([0, 0, 1, 1])
        ax1.set_ylabel("Error")
        ax1.set_xlabel("K")
        ax1.set_yticks(np.arange(0, 1, 0.05))
        ax1.set_xticks(np.arange(0, 50, 2))
        title = "Decision Tree Classifier Test and Train Error vs Depth K for " + str(math.ceil(train_ratio * 100))
        title = title + "/" + str((100 - math.ceil(train_ratio * 100))) + " Train/Test Split"
        ax1.set_title(title)
        plt.plot(K, train_error, 'g--', K, test_error, 'r--', linestyle='solid', linewidth=2, markersize=12)
        green_patch = mpatches.Patch(color='green', label='Train error')
        red_patch = mpatches.Patch(color='red', label='Test error')
        plt.legend(handles=[green_patch, red_patch])
        plt.show(block=True)
        plt.interactive(False)


if __name__ == __main__():
    __main__()
