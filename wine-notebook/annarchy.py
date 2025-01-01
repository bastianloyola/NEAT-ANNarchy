import numpy as np
import random as rd
import scipy.sparse
from neuronmodel import IZHIKEVICH

from ANNarchy import *

IZHIKEVICH = Neuron(  #I = 20
    parameters="""
        a = 0.02 : population
        b = 0.2 : population
        c = -65.0 : population
        d = 8.0 : population
        I = 0.0
        tau_I = 10.0 : population
    """,
    equations="""
        dv/dt = 0.04*v*v + 5*v + 140 - u + I + g_exc - g_inh : init=-65
        tau_I * dg_exc/dt = -g_exc
        tau_I * dg_inh/dt = -g_inh
        du/dt = a*(b*v - u) : init=-14.0
    """,
    spike="v >= 30.0",
    reset="v = c; u += d"
)

def snn(n_inputs, n_outputs, n, i, matrix, inputWeights, trial, genome_id):
    try:
        clear()
        # Create a population with n neurons
        pop = Population(geometry=n, neuron=IZHIKEVICH)
        # Create projection from neurons to itself
        proj = Projection(pre=pop, post=pop, target='exc')
        # Verify the size of the matrix
        if matrix.size == 0:
            raise ValueError('matrix is empty')
        # lil_matrix scipy nxn with values from matrix
        lil_matrix = scipy.sparse.lil_matrix((int(n), int(n)))
        n_rows = matrix.shape[0]
        n_cols = matrix.shape[1]
        lil_matrix[:n_rows, :n_cols] = matrix
        proj.connect_from_sparse(lil_matrix)
        name = 'annarchy/annarchy-' + str(int(trial)) + '/annarchy-' + str(int(i))
        compile(directory=name, clean=False, silent=True)
        M = Monitor(pop, ['spike', 'v'])
        input_index = []
        output_index = []
        n_inputs = int(n_inputs)
        n_outputs = int(n_outputs)
        for i in range(n_inputs):
            input_index.append(i)
        for i in range(n_inputs, n_outputs + n_inputs):
            output_index.append(i)
        # Verify size of inputWeights
        if inputWeights.size == 0:
            raise ValueError('inputWeights is empty')

        fit = fitness(pop, M, input_index, output_index, inputWeights, genome_id)
        return fit
    except Exception as e:
        # Capture the error and print it
        print('Error en annarchy:', e)

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
