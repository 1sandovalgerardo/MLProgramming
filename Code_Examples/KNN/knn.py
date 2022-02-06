import numpy as np
import operator

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0,0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# The psuedo code for the KNN algorithm
"""
For every point in our dataset:

    calculate the distance between inX and the current point
    sort the distances in increasing order
    take k items with lowest distances to inX
    find the majority class among these items
    return the majority class as our prediction for the class of inX
"""

# k-Nearest Neighbors algorithm
    
def classifyZero(inX, data, labels, k):
    '''inX is the point of interest'''
    data_set_size = data.shape[0]
    diff = np.tile(inX, (data_set_size, 1)) - data
    sq_diff = diff**2 
    sum_diff_sqrd = sq_diff.sum(axis=1) 
    sqrt_sum_of_diff_sqrd = sum_diff_sqrd**0.5
    # Calculate distance from inX
    distances_from_x = sqrt_sum_of_diff_sqrd
    # Returns a list where the position contains the position of the value at that rank.
    sorted_distances = distances_from_x.argsort()
    class_count = {}
    for i in range(k):
        vote_i_lable = labels[sorted_distances[i]]
        class_count[vote_i_lable] = class_count.get(vote_i_lable, 0)+1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    number_of_lines = len(arrayOLines)
    return_matrix = np.zeros((number_of_lines, 3))
    class_label_vector = []
    index = 0
    for line in array_of_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_matrix[index, :] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_matrix, class_label_vector

