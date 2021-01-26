from math import sqrt
from random import seed, randrange
from csv import reader
"""
    Iris Flower Species Dataset
    In this tutorial we will use the Iris Flower Species Dataset.
    The Iris Flower Dataset involves predicting the flower species given measurements of iris flowers.
    It is a multiclass classification problem. The number of observations for each class is balanced. 
    There are 150 observations with 4 input variables and 1 output variable. The variable names are as follows:

    *Sepal length in cm.
    *Sepal width in cm.
    *Petal length in cm.
    *Petal width in cm.
    *Class
    
    This k-Nearest Neighbors tutorial is broken down into 3 parts:

    Step 1: Calculate Euclidean Distance.
    Step 2: Get Nearest Neighbors.
    Step 3: Make Predictions
"""
 
# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
 
# convert string colum or discriptor columns to float
# strip pour enlever l'espace si y'en a
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
#convert string column or class colum to integer
#permet d'enlever les redandances des classes
"""
        Iris-versicolor = 0
        Iris-setosa = 1
        Iris-virginica = 2
    """
# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
		print("[%s] => %d" % (value,i))
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Find the min and max values for each column
"""
        [4.3, 7.9]
        [2.0, 4.4]
        [1.0, 6.9]
        [0.1, 2.5]
        [0, 2]
    """
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
            #eliminer les classer pour tester
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores
  

# Step 1: Calculate Euclidean Distance.
# calculate the Euclidean distance between tow vectors
# la dernière colone est l'output on la néglige
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
 


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors
 
#Step 2: Get Nearest Neighbors.
# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors


#Step 3: Make Predictions
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train,test_row,num_neighbors)
    # les classes soit 0,1,2
    output_values = [row[-1] for row in neighbors]
    # il appartient à la classe des voision qui apparait le plus 
    prediction = max(set(output_values),key=output_values.count)
    return prediction

 # kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
	predictions = list()
	for row in test:
		output = predict_classification(train, row, num_neighbors)
		predictions.append(output)
	return(predictions)


# Test the kNN on the Iris Flowers dataset
seed(1)
filename = 'iris.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
num_neighbors = 5
# define a new record
row = [2.7,2.9,4.2,1.3]
# predict the label
label = predict_classification(dataset, row, num_neighbors)
print('Data=%s, Predicted: %s' % (row, label))

# evaluate algorithm
# n_folds = 5
# num_neighbors = 5
# scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors)
# print('Scores: %s' % scores)
# print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))