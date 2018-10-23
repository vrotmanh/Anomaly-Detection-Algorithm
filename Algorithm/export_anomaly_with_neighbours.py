from knn import knn, neighbours_of_index
from helper import find_anomalies, get_input
import sys
import csv

def export_anomaly_and_neighbours(anomaly_index, neighbours_indices):
    with open('anomaly_neighbours/'+str(anomaly_index)+'.csv', 'w') as csvfile:
        fieldnames = ['id','mean radius', 'mean texture',
                              'mean perimeter', 'mean area',
                              'mean smoothness', 'mean compactness',
                              'mean concavity', 'mean concave points',
                              'mean symmetry', 'mean fractal dimension',
                              'radius error', 'texture error',
                              'perimeter error', 'area error',
                              'smoothness error', 'compactness error',
                              'concavity error', 'concave points error',
                              'symmetry error', 'fractal dimension error',
                              'worst radius', 'worst texture',
                              'worst perimeter', 'worst area',
                              'worst smoothness', 'worst compactness',
                              'worst concavity', 'worst concave points',
                              'worst symmetry', 'worst fractal dimension']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in neighbours_indices:
            datapoint_input = get_input(i)
            writer.writerow({
                'id': i,
                'mean radius': datapoint_input[0], 
                'mean texture': datapoint_input[1], 
                'mean perimeter': datapoint_input[2], 
                'mean area': datapoint_input[3], 
                'mean smoothness': datapoint_input[4], 
                'mean compactness': datapoint_input[5], 
                'mean concavity': datapoint_input[6], 
                'mean concave points': datapoint_input[7], 
                'mean symmetry': datapoint_input[8], 
                'mean fractal dimension': datapoint_input[9], 
                'radius error': datapoint_input[10], 
                'texture error': datapoint_input[11], 
                'perimeter error': datapoint_input[12], 
                'area error': datapoint_input[13],
                'smoothness error': datapoint_input[14], 
                'compactness error': datapoint_input[15], 
                'concavity error': datapoint_input[16], 
                'concave points error': datapoint_input[17], 
                'symmetry error': datapoint_input[18], 
                'fractal dimension error': datapoint_input[19], 
                'worst radius': datapoint_input[20], 
                'worst texture': datapoint_input[21],
                'worst perimeter': datapoint_input[22],
                'worst area': datapoint_input[23],
                'worst smoothness': datapoint_input[24],
                'worst compactness': datapoint_input[25],
                'worst concavity': datapoint_input[26],
                'worst concave points': datapoint_input[27],
                'worst symmetry': datapoint_input[28],
                'worst fractal dimension': datapoint_input[29]
                })

anomaly_index = int(sys.argv[1])
anomalies_indices = find_anomalies()
distances, indices = knn()
neighbours_indices = neighbours_of_index(anomaly_index, distances, indices)
export_anomaly_and_neighbours(anomaly_index, neighbours_indices)


