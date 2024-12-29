import subprocess
import requests


def download_neat():
    url = "https://github.com/bastianloyola/NEAT-ANNarchy/raw/refs/heads/main/NEAT"
    response = requests.get(url)

    if response.status_code == 200:
        with open("NEAT", "wb") as file:
            file.write(response.content)

    # Give execution permissions to the file
    subprocess.run(["chmod", "+x", "NEAT"])



def write_annarchy(neuron_model, func):
    # Read the dynamic content of neuronmodel.py
    with open("neuronmodel.py", "r") as neuron_file:
        neuron_model_text = neuron_file.read()

    # Read the dynamic content of fitness.py
    with open("fitness.py", "r") as func_file:
        func_text = func_file.read()

    # Construct the final content of annarchy.py
    lines = [
        "import numpy as np",
        "import random as rd",
        "import scipy.sparse",
        f"from neuronmodel import {neuron_model}",
        "",
        neuron_model_text.strip(),  # Add the content of neuronmodel.py
        "",
        "def snn(n_inputs, n_outputs, n, i, matrix, inputWeights, trial, genome_id):",
        "    try:",
        "        clear()",
        "        # Create a population with n neurons",
        f"        pop = Population(geometry=n, neuron={neuron_model})",
        "        # Create projection from neurons to itself",
        "        proj = Projection(pre=pop, post=pop, target='exc')",
        "        # Verify the size of the matrix",
        "        if matrix.size == 0:",
        "            raise ValueError('matrix is empty')",
        "        # lil_matrix scipy nxn with values from matrix",
        "        lil_matrix = scipy.sparse.lil_matrix((int(n), int(n)))",
        "        n_rows = matrix.shape[0]",
        "        n_cols = matrix.shape[1]",
        "        lil_matrix[:n_rows, :n_cols] = matrix",
        "        proj.connect_from_sparse(lil_matrix)",
        "        name = 'annarchy/annarchy-' + str(int(trial)) + '/annarchy-' + str(int(i))",
        "        compile(directory=name, clean=False, silent=True)",
        "        M = Monitor(pop, ['spike', 'v'])",
        "        input_index = []",
        "        output_index = []",
        "        n_inputs = int(n_inputs)",
        "        n_outputs = int(n_outputs)",
        "        for i in range(n_inputs):",
        "            input_index.append(i)",
        "        for i in range(n_inputs, n_outputs + n_inputs):",
        "            output_index.append(i)",
        "        # Verify size of inputWeights",
        "        if inputWeights.size == 0:",
        "            raise ValueError('inputWeights is empty')",
        "",
        f"        fit = {func}(pop, M, input_index, output_index, inputWeights, genome_id)",
        "        return fit",
        "    except Exception as e:",
        "        # Capture the error and print it",
        "        print('Error en annarchy:', e)",
        "",
        func_text.strip()  # Add the content of fitness.py
    ]

    # Write the content to annarchy.py
    with open("annarchy.py", "w", newline='\n') as f:
        f.write("\n".join(lines))
        f.write("\n") 


def get_function():
    #open config file and get the parameter "function"
    with open('config/config.cfg') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('function='):
                return line.split('=')[1].strip()
            

def get_neuron_model():
    #open config file and get the parameter "neuronModel"
    with open('config/config.cfg') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('neuronModel='):
                return line.split('=')[1].strip()
            

def runNEAT(trial, func, neuron_model, procesos, evolutions, population):

    download_neat()
    #open config file and write the parameters
    with open('config/config.cfg', 'a') as f:
        f.write(f'function={func}\n')
        f.write(f'neuronModel={neuron_model}\n')
        f.write(f'process_max={procesos}\n')
        f.write(f'evolutions={evolutions}\n')
        f.write(f'numberGenomes={population}\n')

    #write the annarchy file
    write_annarchy(neuron_model, func)
    fitness = None
    process = subprocess.Popen(["./NEAT", str(trial)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    output, error = process.communicate()

    process.wait()
    

    if process.returncode != 0:
        print("Output:", output)
        print("Error running NEAT:", error)
        print("Return code:", process.returncode)
    else:
        fitness = float(output.strip().split("\n")[-1])
        return fitness
    
def fitness_value(trial):
    with open(f'results/trial-{trial}/info.txt') as f:
        lines = f.readlines()
        fitness = []
        for line in lines:
            if line.startswith('Genome fitness:'):
                fitness.append(float(line.split(': ')[1].strip()))
        return fitness[-1]