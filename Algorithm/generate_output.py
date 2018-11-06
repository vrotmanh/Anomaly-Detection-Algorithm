from helper import Y_LABEL
import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def get_inputs(index, inputs):
	return [i[index] for i in inputs]

def generate_output(patient_index):
	data = []
	with open('anomaly_neighbours/'+str(patient_index)+'.csv') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',')
		for row in spamreader:
			data.append(row)
	patient_input = [float(i) for i in data[1][1:]]
	all_inputs = [[float(j) for j in i[1:]] for i in data[1:]]
	for input_idx in range(30):
		inputs = get_inputs(input_idx, all_inputs)
		std = np.std(inputs)
		mean = np.mean(inputs)
		z = abs((patient_input[input_idx]-mean)/std)
		z = float("{0:.3f}".format(z))
		above_mean = patient_input[input_idx]>=mean
		t1 = "for " + color.DARKCYAN + Y_LABEL[input_idx].title() + color.END + " the patient is "
		t2 = color.PURPLE + str(z) + color.END
		t3 = " standard deviations below the mean"
		if z>1:
			t2 = color.RED + str(z) + color.END
		if above_mean:
			t3 = " standard deviations above the mean"
		print(t1+t2+t3)

generate_output(289)