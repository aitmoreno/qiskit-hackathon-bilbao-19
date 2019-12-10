from qiskit import *
import numpy as np

class QuantumPCA:    
    def __init__(matrix, num_qbits):
        self.ndarray_matrix = matrix
        self.num_qbits = num_qbits

    max_iterations = 100
    accuracy = 0.1
    shots_per_iteration = 8192
    __rho2__ = 0

    def __run_circuit__(quantum_circuit, state_vector):
        quantum_circuit.initialize(state_vector, num_qbits-1)

        quantum_circuit.h(0)
        quantum_circuit.h(1)

        (th1, ph1, lam1) = qiskit.quantum_info.synthesis.two_qubit_decompose.euler_angles_1q(expm(2*1j*np.pi*__rho2__))
        quantum_circuit.cu3(th1, ph1, lam1, 1, 2)

        (th2, ph2, lam2) = qiskit.quantum_info.synthesis.two_qubit_decompose.euler_angles_1q(expm(2*1j*np.pi*__rho2__*2))
        quantum_circuit.cu3(th2, ph2, lam2, 0, 2)

        quantum_circuit.h(0)
        quantum_circuit.crz(-np.pi/2,0,1)
        quantum_circuit.h(1)

        quantum_circuit.measure([0,1,2],[0,1,2])

        results = execute(quantum_circuit, backend=backend, shots=SHOTS_PER_ITERATION).result().get_counts()

        denominator_result = results['111'] + results['011']
        alpha1 = np.sqrt(results['011'] / denominator_result)
        alpha2 = np.sqrt(results['111'] / denominator_result)
        new_state = [alpha1, alpha2]
        
        return new_state

    def execute():
        sigma2 = np.cov(data.values.T)
        rho2 = sigma2 /np.matrix.trace(sigma2)
        eigenvalues,(eigenvector1, eigenvector2)= np.linalg.eigh(rho2)
        eigenvector1.dot(rho2)
        state_vector = [1,0]
        list_states_vector = list()

        for i in range(0, max_iterations):
            quantum_circuit = QuantumCircuit(num_qbits, num_qbits)
            state_vector = __run_circuit__(quantum_circuit, state_vector)
            list_states_vector.append(state_vector)
        
        return list_states_vector
