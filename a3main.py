import numpy as np
import random as random
from math import exp as e
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pprint import pprint as pprint

# load raw data from csv file, strip new lines and format as floats
def load_data (filename):
	data_array = []
	
	with open(filename, "r") as file:
		# remove column headings
		headers = file.readline()
		# for each input pattern
		for line in file:
			# split comma separated data and remove '\n'
			split_line = line[:-1].split(',')
			numeric_list = [float(val) for val in split_line]
			# store numeric data
			data_array.append(numeric_list)
	# return data in numpy array
	return np.array(data_array)

# writes 2D arrays to file as comma separated values
def write_2d_array (array, filename, header):
	# open file for writing
	with open(filename, 'w') as file:
		# write header line
		file.write(header + '\n')
		# format each row
		for row in array:
			line = ''
			# separate values by comma
			for i in range(len(row) - 1):
				line += '{:8.6f},'.format(row[i])
			# end each line with newline
			line += '{:8.6f}\n'.format(row[-1])
			# write line
			file.write(line)

# generate 2D array of random floats
def random_array (shape, lower, upper):
	ret = []
	for j in range(shape[0]):
		ret.append([random.uniform(lower, upper) for i in range(shape[1])])
	return np.array(ret)

def max_net (activations):
	a = activations
	
	# count number of zero elements in an array
	def zero_count(array):
		count = 0
		for element in array:
			if element == 0: count += 1
		return count

	# perform max-net iterations until there is a single non-zero output
	while zero_count(a) < (len(a) - 1):
		# compute outputs
		outputs = [ max(0, a[i] - ((1 / len(a)) * np.sum(np.append(a[:i], a[i+1:])))) for i in range(len(a)) ]
		# outputs become new activations
		a = outputs

	return outputs


