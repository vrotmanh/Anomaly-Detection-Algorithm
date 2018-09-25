import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets

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

find_anomalies()