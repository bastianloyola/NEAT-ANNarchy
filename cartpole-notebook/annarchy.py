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

        fit = cartpole(pop, M, input_index, output_index, inputWeights, genome_id)
        return fit
    except Exception as e:
        # Capture the error and print it
        print('Error en annarchy:', e)

from ANNarchy import *
import gymnasium as gym
import numpy as np


def cartpole(pop,Monitor,input_index,output_index,inputWeights, genome_id):
    env = gym.make("CartPole-v1")
    observation, info = env.reset()
    terminated = False
    truncated = False
    maxInput = inputWeights[1]
    minInput = inputWeights[0]
    #Generate 4 input weights for each input
    np.random.seed(int(genome_id))
    inputWeights = np.random.uniform(minInput,maxInput,4)
    #Number of episodes
    episodes = 100
    h=0
    #Final fitness
    final_fitness = 0

    # Limits for each observation variable
    limits = [
        (-4.8, 4.8),  # Cart position
        (-10.0, 10.0),  # Cart velocity (estimated)
        (-0.418, 0.418),  # Pole angle in radians
        (-10.0, 10.0)  # Pole angular velocity (estimated)
    ]
    def normalize(value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)

    while h < episodes:
        j=0
        returns = []
        actions_done = []
        terminated = False
        truncated = False
        env.reset()
        while not terminated and not truncated:
            #encode observation, 4 values split in 8 neurons (2 for each value), if value is negative the left neuron is activated, if positive the right neuron is activated
            i = 0
            k = 0
            for val in observation:
                if val < 0:
                    #val = normalize(val, limits[k][0], limits[k][1])
                    pop[int(input_index[i])].I = -val*inputWeights[k]
                    pop[int(input_index[i+1])].I = 0
                else:
                    #val = normalize(val, limits[k][0], limits[k][1])
                    pop[int(input_index[i])].I = 0
                    pop[int(input_index[i+1])].I = val*inputWeights[k]
                i += 2
                k += 1
                print("val normalized:",val)
            simulate(50.0)
            spikes = Monitor.get('spike')
            #Output from 2 neurons, one for each action
            output1 = np.size(spikes[output_index[0]])
            output2 = np.size(spikes[output_index[1]])
            #Choose the action with the most spikes
            action = env.action_space.sample()
            if output1 > output2: #left
                action = 0
            elif output1 < output2: #right
                action = 1
            observation, reward, terminated, truncated, info = env.step(action)
            returns.append(reward)
            actions_done.append(action)
            pop.reset()
            Monitor.reset()
            j += 1
        #The fitness is the sum of the rewards for each episode
        final_fitness += np.sum(returns)
        h += 1
    #The final fitness is the mean of the fitness for each episode
    final_fitness = final_fitness/episodes
    env.close()
    return final_fitness
