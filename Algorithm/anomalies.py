from helper import find_anomalies, get_input, get_all_data_results, Y_LABEL
import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    
    plt.figure(input_index)
    plt.scatter(prediction_not_anomalies,input_not_anomalies,color='blue')
    plt.scatter(prediction_anomalies,input_anomalies,color='red')

    # Setup Graph
    anomaly_legend = mpatches.Patch(color='red', label='Anomaly')
    normal_legend = mpatches.Patch(color='blue', label='Not Anomaly')
    plt.legend(handles=[anomaly_legend, normal_legend])
    plt.xlabel('Probability that tumor is Benign') 
    plt.ylabel(Y_LABEL[input_index].title())
    plt.savefig("Inputs_vs_anomalies_graphs/"+Y_LABEL[input_index].title())
    plt.close()

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

export_anomalies()