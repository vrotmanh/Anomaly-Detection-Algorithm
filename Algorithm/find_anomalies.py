import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
import csv

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

def get_input(index):
    test_data = np.load('./inputs/logistic_regression/testing-logistic_regression-999.npy').tolist()
    train_data = np.load('./inputs/logistic_regression/training-logistic_regression-999.npy').tolist()
    all_data = train_data + test_data
    return all_data[index]

def get_predictions(data):
    predictions = []
    for i in data[0]:
        if i<0.5:
            predictions.append(0)
        else:
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

def find_anomalies():
    (train_data, test_data) = load_data()
    all_data = [train_data[0] + test_data[0], train_data[1] + test_data[1], train_data[2] + test_data[2], train_data[3] + test_data[3]]

    #Predictions
    predictions = get_predictions(all_data)

    #Labels
    actual_labels = get_actual_labels(all_data)

    #Anomalies
    anomalies = []
    for index in range(len(predictions)):
        if predictions[index] != actual_labels[index]:
            anomalies.append(index)

    return(anomalies)

def export_anomalies():
    anomalies = find_anomalies()
    with open('anomalies.csv', 'w') as csvfile:
        fieldnames = ['mean radius', 'mean texture',
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
        for i in anomalies:
            anomaly_input = get_input(i)
            writer.writerow({
                'mean radius': anomaly_input[0], 
                'mean texture': anomaly_input[1], 
                'mean perimeter': anomaly_input[2], 
                'mean area': anomaly_input[3], 
                'mean smoothness': anomaly_input[4], 
                'mean compactness': anomaly_input[5], 
                'mean concavity': anomaly_input[6], 
                'mean concave points': anomaly_input[7], 
                'mean symmetry': anomaly_input[8], 
                'mean fractal dimension': anomaly_input[9], 
                'radius error': anomaly_input[10], 
                'texture error': anomaly_input[11], 
                'perimeter error': anomaly_input[12], 
                'area error': anomaly_input[13],
                'smoothness error': anomaly_input[14], 
                'compactness error': anomaly_input[15], 
                'concavity error': anomaly_input[16], 
                'concave points error': anomaly_input[17], 
                'symmetry error': anomaly_input[18], 
                'fractal dimension error': anomaly_input[19], 
                'worst radius': anomaly_input[20], 
                'worst texture': anomaly_input[21],
                'worst perimeter': anomaly_input[22],
                'worst area': anomaly_input[23],
                'worst smoothness': anomaly_input[24],
                'worst compactness': anomaly_input[25],
                'worst concavity': anomaly_input[26],
                'worst concave points': anomaly_input[27],
                'worst symmetry': anomaly_input[28],
                'worst fractal dimension': anomaly_input[29]
                })

export_anomalies()