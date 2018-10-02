import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
import matplotlib.patches as mpatches
import csv

PARAM_mean_radius = 0
PARAM_mean_texture = 1
PARAM_mean_perimeter = 2
PARAM_mean_area = 3
PARAM_mean_smoothness = 4
PARAM_mean_compactness = 5
PARAM_mean_concavity = 6
PARAM_mean_concave_points = 7
PARAM_mean_symmetry = 8
PARAM_mean_fractal_dimension = 9
PARAM_radius_error = 10
PARAM_texture_error = 11
PARAM_perimeter_error = 12
PARAM_area_error = 13
PARAM_smoothness_error = 14
PARAM_compactness_error = 15
PARAM_concavity_error = 16
PARAM_concave_points_error = 17
PARAM_symmetry_error = 18
PARAM_fractal_dimension_error = 19
PARAM_worst_radius = 19
PARAM_worst_texture = 20
PARAM_worst_perimeter = 21
PARAM_worst_area = 22
PARAM_worst_smoothness = 23
PARAM_worst_compactness = 24
PARAM_worst_concavity = 25
PARAM_worst_concave_points = 26
PARAM_worst_symmetry = 27
PARAM_worst_fractal_dimension = 29

Y_LABEL = ['mean radius', 'mean texture',
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

def find_anomalies():
    all_data = get_all_data_results()

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

def plot_inputs_vs_anomalies(input_index):
    all_data = get_all_data_results()[0]
    anomalies_indexes = find_anomalies()
    input_not_anomalies = []
    prediction_not_anomalies = []
    input_anomalies = []
    prediction_anomalies = []
    for i in range(len(all_data)):
        if i not in anomalies_indexes:
            input_not_anomalies.append(get_input(i)[input_index])
            prediction_not_anomalies.append(all_data[i])
        else:
            input_anomalies.append(get_input(i)[input_index])
            prediction_anomalies.append(all_data[i])
    
    plt.scatter(prediction_not_anomalies,input_not_anomalies,color='blue')
    plt.scatter(prediction_anomalies,input_anomalies,color='red')

    # Setup Graph
    anomaly_legend = mpatches.Patch(color='red', label='Anomaly')
    normal_legend = mpatches.Patch(color='blue', label='Not Anomaly')
    plt.legend(handles=[anomaly_legend, normal_legend])
    plt.xlabel('Probability that tumor is Benign') 
    plt.ylabel(Y_LABEL[input_index].title())
    plt.savefig("Inputs_vs_anomalies_graphs/"+Y_LABEL[input_index].title())


def export_anomalies():
    anomalies = find_anomalies()
    with open('anomalies.csv', 'w') as csvfile:
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
        for i in anomalies:
            anomaly_input = get_input(i)
            writer.writerow({
                'id': i,
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

for i in range(30):
    plot_inputs_vs_anomalies(i)