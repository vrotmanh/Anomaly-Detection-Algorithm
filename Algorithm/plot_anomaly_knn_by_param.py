from helper import Y_LABEL
import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_inputs_vs_anomaly_with_knn(anomaly, input_index):
    data = []
    with open('anomaly_neighbours/'+str(anomaly)+'.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            data.append(row)
    anomaly_input = float(data[1][input_index+1])
    neighbour_inputs = []
    for i in data[2:]:
        neighbour_inputs.append(float(i[input_index+1]))

    plt.figure(input_index)
    plt.scatter(neighbour_inputs, [1]*len(neighbour_inputs),color='blue')
    plt.scatter(anomaly_input, 1,color='red')

    # Setup Graph
    anomaly_legend = mpatches.Patch(color='red', label='Anomaly')
    normal_legend = mpatches.Patch(color='blue', label='Not Anomaly')
    plt.legend(handles=[anomaly_legend, normal_legend])
    plt.ylabel(Y_LABEL[input_index].title())
    plt.savefig("anomaly_neighbours/graphs/"+str(anomaly)+"/"+Y_LABEL[input_index].title())
    plt.close()

for i in range(30):
    plot_inputs_vs_anomaly_with_knn(341, i)