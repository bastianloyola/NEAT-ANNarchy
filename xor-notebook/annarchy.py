import numpy as np
import random as rd
import scipy.sparse
from neuronmodel import LIF

from ANNarchy import *

LIF = Neuron(  #I = 75
    parameters = """
    tau = 50.0 : population
    I = 0.0
    tau_I = 10.0 : population
    """,
    equations = """
    tau * dv/dt = -v + g_exc - g_inh + (I-65) : init=0
    tau_I * dg_exc/dt = -g_exc
    tau_I * dg_inh/dt = -g_inh
    """,
    spike = "v >= -40.0",
    reset = "v = -65"
)

def snn(n_inputs, n_outputs, n, i, matrix, inputWeights, trial, genome_id):
    try:
        clear()
        # Create a population with n neurons
        pop = Population(geometry=n, neuron=LIF)
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

        fit = xor(pop, M, input_index, output_index, inputWeights, genome_id)
        return fit
    except Exception as e:
        # Capture the error and print it
        print('Error en annarchy:', e)

from ANNarchy import *
import numpy as np

def xor(pop,Monitor,input_index,output_index,inputWeights, genome_id):
    Monitor.reset()
    entradas = [(0, 0), (0, 1), (1, 0), (1, 1)]
    fitness = 0
    for entrada in entradas:
        for i, val in zip(input_index, entrada):
            if val == 1:
                pop[int(i)].I = 15.1*inputWeights[i]
            else:
                pop[int(i)].I = 0
        simulate(10.0)
        spikes = Monitor.get('spike')
        #print("spikes: ",spikes)
        #Get the output
        output = []
        for i in output_index:
            output.append(sum(spikes[i]))
        #print("output: ",output)
        decode_output = 0
        if output[0] <= output[1]:
            decode_output = 1

        pop.reset()
        Monitor.reset()
        #comparar las entradas y la salida esperada con el output
        if entrada[0] ^ entrada[1] == decode_output:
            fitness += 1
    return fitness
