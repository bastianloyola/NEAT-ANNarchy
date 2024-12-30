
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
