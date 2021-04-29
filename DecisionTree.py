import math
import numpy as np


def getNumberOfImagesInEachClass(image_matrix):
    num_images_in_each_class = dict()
    for row in image_matrix:
        label = row[-1]
        if label not in num_images_in_each_class:
            num_images_in_each_class[label] = 1
        else:
            num_images_in_each_class[label] += 1

    return num_images_in_each_class


class DecisionNode(object):
    def __init__(self, feature, threshold, left_child, right_child, image_matrix):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.numLabelsInEachClass = getNumberOfImagesInEachClass(image_matrix)


class LabelNode(object):
    def __init__(self, image_matrix):
        self.numLabelsInEachClass = getNumberOfImagesInEachClass(image_matrix)


class DecisionTree(object):
    """
    features - vocab words in euclidean space
    values for features - frequencies of each vocab word

    Partition Method:
    1. determine best feature to split on - best word to split on - maximizes information gain
    2. At each level, we iterate through each feature to find the feature F and threshold T that
       maximally reduces uncertainty

            argmax over (F,T) of u(C) - (p_L * u(C_L) + p_R * u(C_R))

              p_L = fraction of emails going in left child
              p_R = fraction of emails going in right child
    """
    def __init__(self, image_matrix, max_depth):
        self.root = self.buildDecisionTree(image_matrix, 0, max_depth)

    def buildDecisionTree(self, image_matrix, currentDepth, max_depth):

        if currentDepth == max_depth:
            return LabelNode(image_matrix)
        # At each node, we iterate through all features, and find the feature F
        # and threshold T that maximizes information gain and maximally reduces uncertainty
        information_gain, feature, threshold = self.findBestFeatureAndThreshold(image_matrix)

        # Base case - no information gain from partitioning - leaf
        if information_gain == 0:
            return LabelNode(image_matrix)

        # there is information gain, partition emails based on feature and threshold
        left, right = self.partition(image_matrix, feature, threshold)

        # recursively build left and right subtrees
        left_child = self.buildDecisionTree(left, currentDepth + 1, max_depth)
        right_child = self.buildDecisionTree(right, currentDepth + 1, max_depth)

        return DecisionNode(feature, threshold, left_child, right_child, image_matrix)

    def calculateInformationGain(self, left, right, current_uncertainty):
        """
        The uncertainty of the starting node, minus the weighted impurity of
        two child nodes.

                        u(C) - (p_L * u(C_L) + p_R * u(C_R))

                     p_L = fraction of emails going in left child
                     p_R = fraction of emails going in right child
        """
        p_L = float(len(left)) / (len(left) + len(right))
        p_R = float(len(right)) / (len(left) + len(right))
        return current_uncertainty - p_L * self.GiniIndex(left) - p_R * self.GiniIndex(right)

    def GiniIndex(self, image_matrix):

        # get number of images in each label in this cell
        label_counts_dict = getNumberOfImagesInEachClass(image_matrix)
        total_images = image_matrix.shape[0]

        # p_y = fraction of training data labeled y in this cell
        # uncertainty = 1 - ( sum of all (p_y)^2 )
        uncertainty = 1
        for i in range(10):
            if i in label_counts_dict:
                p_y = label_counts_dict[i] / total_images
                uncertainty = uncertainty - math.pow(p_y, 2)

        return uncertainty

    def findBestFeatureAndThreshold(self, image_matrix):
        """
        We iterate through each feature to find the feature F and
        threshold T that maximally reduces uncertainty
        """

        best_information_gain = 0
        best_feature = None
        best_threshold = -1
        current_uncertainty = self.GiniIndex(image_matrix)

        # iterate through all features
        for pixel in range(784):
            # get distinct (thresholds) for the feature vocab word
            frequencies = set(val for val in image_matrix[:,pixel])

            for threshold in frequencies:  # for each value

                # try splitting the dataset
                left, right = self.partition(image_matrix, pixel, threshold)

                if len(left) == 0 or len(right) == 0:
                    # (feature, threshold) combination does not split data
                    continue

                # Calculate the information gain from this split
                information_gain = self.calculateInformationGain(left, right, current_uncertainty)

                # update best (feature, threshold) combination
                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    best_feature = pixel
                    best_threshold = threshold

        return best_information_gain, best_feature, best_threshold

    def partition(self, image_matrix, pixel, threshold):
        """
        feature - vocab word to split on
        threshold - frequency threshold
        """
        left = []
        right = []
        for row in image_matrix:
            # find pixel and corresponding value, to compare with threshold
            pixel_val = row[pixel]

            if pixel_val <= threshold:
                left.append(row)
            else:
                right.append(row)
        return np.asarray(left), np.asarray(right)

    def classifyDecisionTree(self, image_vector, node, currentDepth, K):

        # Base case: we've reached a label or reached stop depth- give the classification
        if isinstance(node, LabelNode) or currentDepth == K:
            best_label = -1
            max_count = -1
            for label in node.numLabelsInEachClass.keys():
                if node.numLabelsInEachClass[label] > max_count:
                    best_label = label
                    max_count = node.numLabelsInEachClass[label]

            return best_label

        pixel_val = image_vector[node.feature]

        if pixel_val <= node.threshold:
            return self.classifyDecisionTree(image_vector, node.left_child, currentDepth + 1, K)
        return self.classifyDecisionTree(image_vector, node.right_child, currentDepth + 1, K)

    def classify(self, image_vector, K):
        return self.classifyDecisionTree(image_vector, self.root, 0, K)
