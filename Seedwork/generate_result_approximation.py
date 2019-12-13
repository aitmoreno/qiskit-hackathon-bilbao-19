import numpy as np
from scipy.linalg import expm
from qiskit import *
import pandas as pd
from qiskit.visualization import *
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/ibonreinoso/qiskit-hackathon-bilbao-19/master/DAX_PERFORMANCE_INDEX.csv"
data = pd.read_csv(url, sep=';')

data = data.drop(['wkn_500340'], axis = 1)
data = data.loc[:,['wkn_515100', 'wkn_575200']]
print(data)

sigma2 = np.cov(data.values.T)
rho2 = sigma2 /np.matrix.trace(sigma2)
print(rho2)

eigenvalues,(eigenvector1, eigenvector2)= np.linalg.eigh(rho2)
print(eigenvalues, eigenvector1, eigenvector2)

eigenvector1.dot(rho2)
eigenvector1 * eigenvalues[0]

NUM_QUBITS = 3
NUM_ITERATION = 50
SHOTS_PER_ITERATION = 8192
backend = BasicAer.get_backend('qasm_simulator')
state_vector = [1,0]
list_states_vector = list()

for i in range(0, NUM_ITERATION):
    quantum_circuit = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
    quantum_circuit.initialize(state_vector, NUM_QUBITS-1)
    quantum_circuit.h(0)
    quantum_circuit.h(1)
    (th1, ph1, lam1) = qiskit.quantum_info.synthesis.two_qubit_decompose.euler_angles_1q(expm(2*1j*np.pi*rho2))
    quantum_circuit.cu3(th1, ph1, lam1, 1, 2)
    (th2, ph2, lam2) = qiskit.quantum_info.synthesis.two_qubit_decompose.euler_angles_1q(expm(2*1j*np.pi*rho2*2))
    quantum_circuit.cu3(th2, ph2, lam2, 0, 2)
    quantum_circuit.h(0)
    quantum_circuit.crz(-np.pi/2,0,1)
    quantum_circuit.h(1)
    quantum_circuit.measure([0,1,2],[0,1,2])
    results = execute(quantum_circuit, backend=backend, shots=SHOTS_PER_ITERATION).result().get_counts()
    result111 = results.get('111', 0)
    result011 = results.get('011', 0)
    denominator_result = result111 + result011
    alpha1 = np.sqrt(result011 / denominator_result)
    alpha2 = np.sqrt(result111 / denominator_result)
    new_state = [alpha1, alpha2]
    # check the acc 
    state_vector = new_state
    list_states_vector.append(state_vector)
    plt.plot(np.array(list_states_vector).T[0])
    plt.savefig('Seedwork/plots/result_approximation_' + str(i))
