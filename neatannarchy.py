import subprocess
import requests
import json
import matplotlib.pyplot as plt
import numpy as np


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
    

bestFile = 'best0.txt'
configFile = 'config.cfg'
infoFile = 'info.txt'
operatorsFile = 'operadores.txt'
resultsFile = 'results.txt'

def readBestFile(file, nodes, conexions):
    with open(file, 'r') as best:
        for line in best:
            line = line.strip().split(';')
            nodes.add(int(line[0]))
            nodes.add(int(line[1]))
            conexions.add((int(line[0]), int(line[1])))

def readConfigFile(file, configutarion):
    with open(file, 'r') as config_file:
        for line in config_file:
            line = line.strip().split('=')
            if len(line) == 2:
                key = line[0].strip()
                value = line[1].strip()
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                configutarion[key] = value

def readOperatorsFile(file, operators):
    with open(file, 'r') as op:
        for line in op:
            if line == '\n' or 'Generacion' in line:
                continue
            if 'Total' in line:
                break
            line = line.strip().split()
            if len(line) == 3:
                key = line[1].strip()[:-1]
                num = int(line[2].strip())
                if key == 'mutacionPeso': key = 'weightMutations'
                elif key == 'mutacionPesoInput': key = 'inputWeightMutations'
                elif key == 'agregarNodos': key = 'addNodes'
                elif key == 'agregarLinks': key = 'addConexion'
                elif key == 'reproducirInter': key = 'reproduceInter'
                elif key == 'reproducirIntra': key = 'reproduceIntra'
                elif key == 'reproducirMuta': key = 'reproduceMutation'
                if key in operators:
                    operators[key].append(num)

def readInfoFile(file, info):
    gen = 0
    etapa = ''
    n= 0
    with open(file, 'r') as info_file:
        for line in info_file:
            if 'Evaluation' in line:
                etapa = 'Evaluation'
                continue
            elif 'Eliminate' in line:
                etapa = 'Eliminate'
                continue
            elif ' Mutation' in line:
                etapa = 'Mutation'
                continue
            elif 'Reproduce' in line:
                etapa = 'Reproduce'
                continue
            elif 'Speciation' in line:
                etapa = 'Speciation'
                continue
            elif 'Best Genome' in line:
                etapa = 'BestGenome'
                continue
            elif 'Generation' in line:
                n += 1
                line = line.strip().split()
                gen = int(line[2])
                info['redroduced'].append([0, 0, 0])
                info['species'].append([])
                info['bestGenome'].append([])
                info['eliminated'].append(0)
                continue
            elif line == '\n':
                continue

            elif etapa == 'Eliminate':
                if 'eliminated' in line:
                    info['eliminated'][gen] += 1
                continue
            elif etapa == 'Mutation':
                continue
            elif etapa == 'Reproduce':
                line = line.strip().split()
                if len(line) < 3:
                    continue
                if 'reproduceInterSpecies' in line[1]:
                    info['redroduced'][gen][0] += int(line[2])
                elif 'reproduceNonInterSpecies' in line[1]:
                    info['redroduced'][gen][1] += int(line[2])
                elif 'reproduceMutations' in line[1]:
                    info['redroduced'][gen][2] += int(line[2])
                continue
            elif etapa == 'Speciation':
                line = line.strip().split()
                if 'Species' in line[0]:
                    info['species'][gen].append(int(line[3]))
                continue
            elif etapa == 'BestGenome':
                line = line.strip().split(':')
                num = float(line[1]) if 'fitness' in line[0] else int(line[1])
                info['bestGenome'][gen].append(num)
                continue


def readResultsFile(file, info):
    with open(file, 'r') as results_file:
        for line in results_file:
            if 'Genome fitness' in line:
                line = line.strip().split()
                num = float(line[2])
                info['bestGenome'][-1].append(0)
                info['bestGenome'][-1].append(0)
                info['bestGenome'][-1].append(num)


def information(folder):
    file = folder
    nodes = set()
    conexions = set()
    configutarion = dict()
    operators = {'weightMutations': [], 'inputWeightMutations': [], 'addNodes': [], 'addConexion': [], 'reproduceInter': [], 'reproduceIntra': [], 'reproduceMutation': []}
    info = {'eliminated': [], 'redroduced': [], 'species': [], 'bestGenome': []}

    readConfigFile(file + configFile, configutarion)
    #print('Configuracion: \n', configutarion)
    readOperatorsFile(file + operatorsFile, operators)
    #print('operators: \n', operators)
    readBestFile(file + bestFile, nodes, conexions)
    #print('Best: \n', nodes, conexions)
    readInfoFile(file + infoFile, info)
    #print('Info: \n', info)
    readResultsFile(file + resultsFile, info)

    #print('Info: ')
    #for i in range(configutarion.get('evolutions', 0)):
        #print(i)
        #print('--> eliminated: ', info['eliminated'][i])
        #print('--> redroduced: ', info['redroduced'][i])
        #print('--> Species: ', info['species'][i])
        #print('--> BestGenome: ', info['bestGenome'][i])

    outputFile = file+'output.json'

    data_to_save = {
        'Configuracion': configutarion,
        'operators': operators,
        'nodes': list(nodes),
        'conexions': list(conexions),
        'Info': {
            'eliminated': info['eliminated'],
            'redroduced': info['redroduced'],
            'Species': info['species'],
            'BestGenome': info['bestGenome']
        }
    }

    with open(outputFile, 'w') as f:
        json.dump(data_to_save, f, indent=4, separators=(", ", ": "))

    print(f"Datos guardados en {outputFile}")




def plot_information(trial):

    folder = f'results/trial-{str(trial)}/'

    information(folder)
    
    outputFile = folder + 'output.json'
    loaded = json.load(open(outputFile))
    info = loaded['Info']
    operators = loaded['operators']

    generaciones = range(len(info['eliminated']))

    fig1 = plt.figure(figsize=(12, 12))
    gs1 = fig1.add_gridspec(4, 1)
    fig1.suptitle('Satistics of Generations - Page 1', fontsize=20)

    # Gráfico de eliminated
    ax1 = fig1.add_subplot(gs1[0, 0])
    ax1.plot(generaciones, info['eliminated'], label='eliminated', color='red')
    ax1.set_title('Eliminated per Generation')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Amount')
    ax1.legend()

    # Gráfico de redroduced
    ax2 = fig1.add_subplot(gs1[1, 0])
    redroduced_inter, redroduced_intra, redroduced_muta = zip(*info['redroduced'])
    ax2.plot(generaciones, redroduced_inter, label='Inter-specie', color='blue')
    ax2.plot(generaciones, redroduced_intra, label='Intra-specie', color='green')
    ax2.plot(generaciones, redroduced_muta, label='Mutation', color='purple')
    ax2.set_title('Redroduced per Generation')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Amount')
    ax2.legend()

    # Gráfico de especies
    ax3 = fig1.add_subplot(gs1[2, 0])
    ax3.plot(generaciones, [len(species) for species in info['Species']], label='Species', color='orange')
    ax3.set_title('Number of Species per Generation')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Amount de Especies')
    ax3.legend()

    # Gráfico del mejor genoma
    ax4 = fig1.add_subplot(gs1[3, 0])
    fitness = [best_genome[2] for best_genome in info['BestGenome']]
    ax4.plot(generaciones, fitness, label='Fitness', color='cyan')
    ax4.set_title('Fitness of the Best Genome per Generation')
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Fitness')
    ax4.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig(folder + 'page1_statistics.png')

    # Segunda página combinada de operators
    fig2 = plt.figure(figsize=(12, 12))
    gs2 = fig2.add_gridspec(1, 1)
    fig2.suptitle('Statistics per Generation - Page 2', fontsize=20)

    # Gráfico combinado de operators
    ax_op = fig2.add_subplot(gs2[0, 0])
    ax_op.plot(generaciones, operators['weightMutations'], label='Weight Mutation', color='magenta')
    ax_op.plot(generaciones, operators['inputWeightMutations'], label='Input Weight Mutation', color='gray')
    ax_op.plot(generaciones, operators['addNodes'], label='Add Nodes', color='blue')
    ax_op.plot(generaciones, operators['addConexion'], label='Add Conexions', color='green')
    ax_op.set_title('Operators per Generation')
    ax_op.set_xlabel('Generation')
    ax_op.set_ylabel('Amount')
    ax_op.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(folder + 'page2_operators.png')

    # Tercera página para la distribución de genomas por especie
    fig3 = plt.figure(figsize=(12, 12))
    gs3 = fig3.add_gridspec(1, 1)
    fig3.suptitle('Genome Distribution per Specie per Generation - Page 3', fontsize=20)

    ax_barras = fig3.add_subplot(gs3[0, 0])
    max_especies_por_generacion = max(len(species) for species in info['Species'])
    genomas_por_especie = []

    for i in range(max_especies_por_generacion):
        especie_data = [gen_species[i] if i < len(gen_species) else 0 for gen_species in info['Species']]
        genomas_por_especie.append(especie_data)

    x = np.arange(len(generaciones))
    bottom = np.zeros(len(generaciones))

    for i, especie_data in enumerate(genomas_por_especie):
        ax_barras.bar(x, especie_data, bottom=bottom, label=f'Specie {i+1}')
        bottom += especie_data

    ax_barras.set_xlabel('Generation')
    ax_barras.set_ylabel('Amount de Genomas')
    ax_barras.legend(title='Species', loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig3.savefig(folder + 'page3_distribution_genomes.png')

    # plt.show()
