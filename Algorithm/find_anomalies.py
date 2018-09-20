import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def load_data(file_index):
    data = np.load('./results/logistic_regression/testing-logistic_regression-'+str(file_index)+'.npy')

    probLabel1 = []
    probLabel2 = []
    for i in range(len(data)):
        probLabel1.append(data[i][0])
        probLabel2.append(data[i][1])

    return (probLabel1, probLabel2)

def plot_malign_vs_benign(file_index):
    data = load_data(file_index)

    plt.scatter(data[0], data[1])
    plt.show()

def plot_input_vs_output(file_index, input_index):
    data = load_data(file_index)
    inputs = np.load('./inputs/logistic_regression/testing-logistic_regression-'+str(file_index)+'.npy')

    inp = []
    for i in range(len(inputs)):
        print(inputs[i][input_index])
        inp.append(inputs[i][input_index])

    plt.scatter(inp, data[0])
    plt.show()

#plot_malign_vs_benign(0)

#plot_input_vs_output(0, 0)
# plot_input_vs_output(0, 1)
plot_input_vs_output(0, 20)