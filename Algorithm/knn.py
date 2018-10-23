import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
import matplotlib.patches as mpatches
import csv
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

def load_data_by_file(file_index, testing=False):
	if testing:
		data = np.load('./results/logistic_regression/testing-logistic_regression-'+str(file_index)+'.npy')
	else:
		data = np.load('./results/logistic_regression/training-logistic_regression-'+str(file_index)+'.npy')

	probLabel1 = []
	probLabel2 = []
	actualLabel1 = []
	actualLabel2 = []
	for i in range(len(data)):
		probLabel1.append(data[i][0])
		probLabel2.append(data[i][1])
		actualLabel1.append(data[i][2])
		actualLabel2.append(data[i][3])

	return [probLabel1, probLabel2, actualLabel1, actualLabel2]

def load_data():
	test_data = load_data_by_file(999, True)
	train_data = [[],[],[],[]]
	for i in range(997, 1000):
		data = load_data_by_file(i)
		train_data[0] = train_data[0] + data[0]
		train_data[1] = train_data[1] + data[1]
		train_data[2] = train_data[2] + data[2]
		train_data[3] = train_data[3] + data[3]
	return(train_data, test_data)

def find_anomalies():
	all_data = get_all_data_results()
	predictions = get_predictions(all_data)
	actual_labels = get_actual_labels(all_data)
	anomalies = []
	for index in range(len(predictions)):
		if predictions[index] != actual_labels[index]:
			anomalies.append(index)

	return(anomalies)

def get_predictions(data):
	predictions = []
	more_than0 = 0
	less_than0 = 0
	for i in data[0]:
		if i<0.5:
			less_than0 +=1
			predictions.append(0)
		else:
			more_than0 +=1
			predictions.append(1)

	return predictions

def get_actual_labels(data):
	labels = []
	for i in data[2]:
		if i<0.5:
			labels.append(0)
		else:
			labels.append(1)
	return labels

def get_all_data_results():
	(train_data, test_data) = load_data()
	return [train_data[0] + test_data[0], train_data[1] + test_data[1], train_data[2] + test_data[2], train_data[3] + test_data[3]]


def get_inputs():
	test_data = np.load('./inputs/logistic_regression/testing-logistic_regression-999.npy').tolist()
	train_data = np.load('./inputs/logistic_regression/training-logistic_regression-999.npy').tolist()
	all_data = train_data + test_data
	return all_data

def knn():
	all_data = get_inputs()
	normalized_data = list(preprocessing.normalize(all_data))
	labels = get_actual_labels(get_all_data_results())
	anomalies_indices = find_anomalies()
	anomalies_inputs = []
	for i in anomalies_indices:
		anomalies_inputs.append(normalized_data[i])
	nbrs = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(normalized_data)
	distances, indices = nbrs.kneighbors(anomalies_inputs)
	return(distances, indices)


def plot_anomalies_vs_neighbors(distances, indices, anomalies_indices):
	all_data = get_inputs()
	predictions = get_all_data_results()[0]
	distances, indices = knn()
	for i in range(len(anomalies_indices)):
		plot_predictions = []
		for j in indices[i]:
			plot_predictions.append(predictions[j])
		plt.figure(i)
		plt.scatter(plot_predictions,distances[i],color='blue')
		plt.scatter([predictions[anomalies_indices[i]]],[0],color='red')
		for k, txt in enumerate(indices[i]):
			plt.annotate(" "+str(txt), (plot_predictions[k], distances[i][k]))
		# Setup Graph
		anomaly_legend = mpatches.Patch(color='red', label='Anomaly')
		normal_legend = mpatches.Patch(color='blue', label='Nearest Neighbors')
		plt.legend(handles=[anomaly_legend, normal_legend])
		plt.xlabel('Probability that tumor is Benign') 
		plt.ylabel("Distance to Anomaly " + str(i))
		plt.savefig("anomalies_knn/"+str(anomalies_indices[i]))
		plt.close()

anomalies_indices = find_anomalies()
distances, indices = knn()
plot_anomalies_vs_neighbors(distances, indices, anomalies_indices)





