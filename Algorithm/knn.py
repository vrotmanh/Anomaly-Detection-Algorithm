from helper import get_inputs, get_actual_labels, get_all_data_results, find_anomalies
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

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





