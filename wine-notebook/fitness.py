
from ANNarchy import *
from sklearn.datasets import load_wine
from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import random as rd

# DATA
# Load the data from sklearn
def load_dataset(dataset_loader):
    data = dataset_loader()
    return np.array(data.data), np.array(data.target)

def normalize_data(data):
    # Normalize the data between [0, 1] using numpy
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

# Load the example dataset
data_X, data_y = load_dataset(load_wine)
# Normalize the data between [0, 1]
data_X_normalized = normalize_data(data_X)

# CODING
# Single-Spike encoding
def single_spike_encoding(data, num_neurons=2):
    encoded_data = []
    step = 1 / num_neurons
    for value in data:
        neuron_idx = min(int(value / step), num_neurons - 1)
        spike_train = [0] * num_neurons
        spike_train[neuron_idx] = 1
        encoded_data.append(spike_train)
    return np.array(encoded_data)

# DECODING
# Voting decoding
def vote_decoding(spikes, n_input, n_output):
    votes = []
    total = n_input + n_output
    for i in range(n_input,total):
        votes.append(len(spikes[i]))
    max_spikes = max(votes)
    index_max = [i for i, x in enumerate(votes) if x == max_spikes]
    
    
    if len(index_max) == 1:
        index = index_max[0]
    else:
        index = rd.choice(index_max)

    return index

# SIMULATION
def simulate_single_spike(data, n_input, neurons_per_input, pop, time=10.0, flag = False):
    simulate(5.0)
    sample = single_spike_encoding(data, neurons_per_input)
    if flag: print("sample: ",sample)
    # For each sample
    if neurons_per_input != 1:
        n = neurons_per_input
        for k in range(n_input):
            if sample[k//n][k%n] == 1:
                pop[k].v = -40.0
            if flag: print("pop[k].v: ",pop[k].v)
    else:
        for k in range(n_input):
            pop[k].v = -40.0 if sample[k] == 1 else 0.0 
    simulate(time)

def bootstrap_data(data_x, data_y,n_bootstrap, n_samples):
    # List to save the generated datasets
    bootstrap_datasets = []
    # Generate the datasets using bootstrapping
    for i in range(n_bootstrap):
        X_bootstrap, y_bootstrap = resample(data_x, data_y, replace=True, n_samples=n_samples)
        bootstrap_datasets.append((X_bootstrap, y_bootstrap))
    return bootstrap_datasets
  
def fitness(pop, M, input_index, output_index, inputWeights, genome_id ,flag=False):
    print("genome_id: ", genome_id)
    n_input = len(input_index)
    n_output = len(output_index)
    subsets = bootstrap_data(data_X_normalized, data_y,50,50)
    print(1)
    total = 0
    n = len(subsets)
    print(2)
    for i in range(n):
        x = subsets[i][0]
        y = subsets[i][1]

        sum = 0
        samples = len(x)
        for j in range(samples):
            neuronas_por_input = 5
            target = y[j]
            simulate_single_spike(x[j], n_input, neuronas_por_input, pop, 10.0, flag)
            spikes = M.get('spike')

            index = vote_decoding(spikes, n_input, n_output)
            if index == target:
                    sum += 1.0

            pop.reset()
            M.reset()
        total += (sum/samples)
    fitness = round(total/n,2)
    print("    fitness: ", fitness)
    return fitness
