import os
from quantuminspire.credentials import load_account, get_token_authentication, get_basic_authentication
from qiskit.circuit.instruction import Instruction
from qiskit.visualization import plot_histogram

from qiskit import BasicAer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import execute

from quantuminspire.qiskit import QI

import numpy as np
import matplotlib.pyplot as plt
import random as rd

# Explanation Code:

# Overall scalar of gate error
scalar = 1
# Qubit used for probability
ERROR_QUBIT = 14
wait_U2 = 100
wait_buffer = 20    # Frequency with which gates can be executed
t1_register = ClassicalRegister(1, name="t1")


def buffered(time):
    return np.ceil(time / wait_buffer) * wait_buffer


gate_time = {
    "z": 0,
    "h": wait_U2,
    "x": 2*wait_U2,
    "y": 2*wait_U2
}


# T1 - and T2 time per qubit in micro seconds
IBM_QX_T = \
    [[74474.82930150885, 19048.7270030086],
     [51548.53511959554, 60191.8134546785],
     [78388.42876617767, 143610.72026264597],
     [75062.16206944322, 56286.2200530015],
     [68076.17160966979, 36410.37911063108],
     [24018.152431061452, 50051.2947180762],
     [75764.22949183788, 68636.04235030044],
     [41072.583314992954, 64071.06508612124],
     [107972.62937569239, 154970.88524657173],
     [45286.354811947604, 63514.46382402567],
     [61542.94883451763, 46163.444102428585],
     [39946.851887638104, 120489.6056753136],
     [5088.04144360129, 9070.650565610016],
     [27441.28124580624, 36642.21813095793],
     [42543.170586979805, 52319.24080728063]]

# U2 gate error rate per qubit
IBM_QX_U2 = \
    [
        0.000428,
        0.002502,
        0.000859,
        0.000451,
        0.001023,
        0.001746,
        0.001098,
        0.001468,
        0.000431,
        0.001336,
        0.002232,
        0.000375,
        0.005269,
        0.002279,
        0.000811
    ]

# measurement error per qubit
IBM_QX_mse = \
    [
        0.0170,
        0.1405,
        0.0140,
        0.0510,
        0.0280,
        0.0425,
        0.0250,
        0.0385,
        0.0570,
        0.0200,
        0.0510,
        0.0235,
        0.1650,
        0.0895,
        0.0405
    ]

IBM_QX_connections = [
    [1, 0],
    [1, 2],
    [2, 3],
    [4, 3],
    [5, 4],
    [5, 6],
    [7, 8],
    [9, 8],
    [9, 10],
    [11, 10],
    [11, 12],
    [13, 12],
    [13, 1],
    [12, 2],
    [11, 3],
    [4, 10],
    [5, 9],
    [6, 8]
]

# value 1: gate time; value 2: error rate
IBM_QX_cx_data = {
    "CX1_0":    [239, 0.0333],
    "CX1_2":    [174, 0.03729],
    "CX2_3":    [261, 0.02804],
    "CX4_3":    [266, 0.01865],
    "CX5_4":    [300, 0.02662],
    "CX5_6":    [300, 0.03524],
    "CX7_8":    [220, 0.02170],
    "CX9_8":    [434, 0.02144],
    "CX9_10":   [300, 0.04064],
    "CX11_10":  [261, 0.03770],
    "CX11_12":  [261, 0.06394],
    "CX13_12":  [300, 0.08875],
    "CX13_1":   [652, 0.08022],
    "CX12_2":   [1043, 0.04094],
    "CX11_3":   [286, 0.02141],
    "CX4_10":   [261, 0.03698],
    "CX5_9":    [348, 0.02989],
    "CX6_8":    [348, 0.02476]
}


def get_cx_data(start, end):
    name = 'CX' + str(start) + '_' + str(end)
    if name not in IBM_QX_cx_data:
        return [4200, 0.1]
    return IBM_QX_cx_data[name]


# returns intersect of two list
def intersect(li1, li2):
    return [val for val in li1 if val in li2]


# returns viable path for CNOT gate following available pairs
def get_cnot_path(control, target):
    paths = [[control]]
    while True:
        new_paths = []
        for path in paths:
            for step in IBM_QX_connections:
                if path[-1] in step:
                    new_path = path.copy()
                    new_path.append(step[1] if step[0] == path[-1] else step[0])
                    if new_path[-1] == target:
                        return new_path
                    new_paths.append(new_path)
        paths = new_paths


def model_cnot(control, target):
    path = get_cnot_path(control, target)
    cnots = []

    # convert path into a list of pairs
    if len(path) <= 1:
        return [path]

    # diagonal up
    for i in range(len(path) - 1, 0, -1):
        cnots.append([path[i - 1], path[i]])

    # diagonal down
    for i in range(1, len(path) - 2):
        cnots.append([path[i], path[i + 1]])

    # diagonal up, again
    for i in range(len(path) - 1, 0, -1):
        cnots.append([path[i - 1], path[i]])

    # partial diagonal down
    for i in range(1, len(path) - 2):
        cnots.append([path[i], path[i + 1]])
    return cnots


def put_random_error(circuit, qubit, error):
    minor_error = (1 - np.sqrt(1 - error)) * scalar
    circuit.rx(angle(minor_error), ERROR_QUBIT)
    circuit.cx(ERROR_QUBIT, qubit)
    circuit.reset(ERROR_QUBIT)

    circuit.rx(angle(minor_error), ERROR_QUBIT)
    circuit.cz(ERROR_QUBIT, qubit)
    circuit.reset(ERROR_QUBIT)


# use time=1234 for a time in microseconds
def put_t1_error(circuit, qubit, chance=None, time=None):
    if chance is None and time is None:
        raise Exception("Please provide an error chance or decay time.")
    elif time is not None:
        chance = (1 - np.exp(-1 * time / IBM_QX_T[qubit][0]))

    rotation = 2 * np.arcsin(np.sqrt(chance))
    circuit.rx(rotation, ERROR_QUBIT)
    circuit.measure(ERROR_QUBIT, t1_register[0])
    circuit.reset(qubit).c_if(t1_register, 1)
    circuit.reset(ERROR_QUBIT)


# places T2 error in circuit
def put_t2_error(circuit, qubit, time):
    circuit.rz(
        np.random.normal(
            0,
            np.sqrt(2 * (1 - np.exp(-time / IBM_QX_T[qubit][1])))
        ),
        qubit
    )


# random value with dev=1 => return -pi_pi
def rand(dev):
    return (rd.random()-0.5) * 2 * np.pi * dev


# generates rotation angle from error chance
def angle(chance):
    return 2 * np.arcsin(np.sqrt(chance))


def put_measure(circuit, fro, to):
    if not isinstance(fro, list):
        put_random_error(circuit, fro, IBM_QX_mse[fro])
    else:
        for q in fro:
            put_random_error(circuit, q, IBM_QX_mse[q])
    circuit.measure(fro, to)


# Insert gates as a dictionary as follows:
# gates = { 'h' : [0, 1, 2], 'cx' : [[3, 4], [5, 6]] }
def put_gates(circuit, gates):
    # individually, each qubit begins without any occupation time
    qubit_time = {i: 0 for i in range(circuit.n_qubits)}
    max_wait = 0
    if 'h' in gates:
        if max_wait < gate_time['h']:
            max_wait = buffered(gate_time['h'])
        circuit.h(gates['h'])
        for i in gates['h']:
            qubit_time[i] += gate_time['h']
            put_random_error(circuit, i, IBM_QX_U2[i])

    if 'x' in gates:
        if max_wait < gate_time['x']:
            max_wait = buffered(gate_time['x'])
        circuit.x(gates['x'])
        for i in gates['x']:
            qubit_time[i] += gate_time['x']
            put_random_error(circuit, i, IBM_QX_U2[i])
            put_random_error(circuit, i, IBM_QX_U2[i])

    if 'y' in gates:
        if max_wait < gate_time['y']:
            max_wait = buffered(gate_time['y'])
        circuit.y(gates['y'])
        for i in gates['y']:
            qubit_time[i] += gate_time['y']
            put_random_error(circuit, i, IBM_QX_U2[i])
            put_random_error(circuit, i, IBM_QX_U2[i])

    if 'z' in gates:
        circuit.z(gates['z'])
        # z gate is free!

    if 's' in gates:
        circuit.s(gates['s'])
        # s gate is free!

    if 'sdg' in gates:
        circuit.sdg(gates['sdg'])
        # sdg gate is free!

    if 't' in gates:
        circuit.z(gates['t'])
        # t gate is free!

    if 'tdg' in gates:
        circuit.z(gates['tdg'])
        # tdg gate is free!

    if 'cx' in gates:
        actual_cx_gates = []
        total_cx_time = 0
        for g in gates['cx']:
            actual_cx_gates.extend(model_cnot(g[0], g[1]))
        for k in range(len(actual_cx_gates)):
            cxg = actual_cx_gates[k]
            if cxg not in IBM_QX_connections:
                # insert Hadamard errors for "inverting" cx gate
                # "fuse" neighbouring hadamard gates to the left
                had_gates = [temp for temp in cxg if temp not in (intersect(cxg, actual_cx_gates[k - 1]) if k > 0 else [])]
                if had_gates:
                    total_cx_time += buffered(gate_time['h'])
                for q in had_gates:
                    qubit_time[q] += gate_time['h']
                    put_random_error(circuit, q, IBM_QX_U2[q])

                # put CX gate
                circuit.cx(cxg[0], cxg[1])

                # model error
                cx_data = get_cx_data(cxg[1], cxg[0])
                total_cx_time += buffered(cx_data[0])

                # introduce error from gate
                put_random_error(circuit, cxg[0], cx_data[1])
                put_random_error(circuit, cxg[1], cx_data[1])

                # account for "busy" time of qubits
                qubit_time[cxg[0]] += cx_data[0]
                qubit_time[cxg[1]] += cx_data[0]
                # "fuse" neighbouring hadamard gates to the right
                had_gates = [temp for temp in cxg if temp not in (intersect(cxg, actual_cx_gates[k + 1]) if k < len(actual_cx_gates) - 1 else [])]
                if had_gates:
                    total_cx_time += buffered(gate_time['h'])
                for q in had_gates:
                    qubit_time[q] += gate_time['h']
                    put_random_error(circuit, q, IBM_QX_U2[q])
            else:
                # put CX gate
                circuit.cx(cxg[0], cxg[1])

                # model error
                cx_data = get_cx_data(cxg[0], cxg[1])
                total_cx_time += buffered(cx_data[0])

                # introduce error from gate
                put_random_error(circuit, cxg[0], cx_data[1])
                put_random_error(circuit, cxg[1], cx_data[1])

                # account for "busy" time of qubits
                qubit_time[cxg[0]] += cx_data[0]
                qubit_time[cxg[1]] += cx_data[0]

        if total_cx_time > max_wait:
            max_wait = total_cx_time

    # now, to account for waiting time:
    for i in range(circuit.n_qubits - 4):
        put_t1_error(circuit, i, time=max_wait - qubit_time[i])
        put_t2_error(circuit, i, max_wait - qubit_time[i])


def put_toffoli(circuit, control1, control2, target):
    put_gates(circuit, {'h': [target]})
    put_gates(circuit, {'cx': [[control2, target]]})
    # t and tdg gates are considered error- and timeless
    circuit.tdg(target)
    put_gates(circuit, {'cx': [[control1, target]]})
    circuit.t(target)
    put_gates(circuit, {'cx': [[control2, target]]})
    circuit.tdg(target)
    put_gates(circuit, {'cx': [[control1, target]]})
    circuit.t([control2, target])
    put_gates(circuit, {'h': [target], 'cx': [[control1, control2]]})
    circuit.t(control1)
    circuit.tdg(control2)
    put_gates(circuit, {'cx': [[control1, control2]]})


# SET UP SIMULATION
q = QuantumRegister(15, 'q')
c0 = ClassicalRegister(5, 'c0')
circuit = QuantumCircuit(q, c0, t1_register, name="Error-modelling circuit")

#############################
# Insert your circuit here! #
#############################
#
# Insert gates as follows:
#
# put_gates(circuit, {
#    'x': [0, 1], 'cx' [[2, 3], [4, 5]], 'h': [6]
# })
#
# Insert toffoli (ccx) gates as follows:
#
# put_toffoli(circuit, control1, control2, target)

put_gates(circuit, {
    'x': [0]
})

###########################################
# Use "put_measure" to place measurements #
###########################################

# Number of measured qubits
n_ms = 1
n_shots = 250

put_measure(circuit, [i for i in range(n_ms)], [i for i in range(n_ms)])

print("Circuit was made! Onto execution...")

# Execute the circuit on the qasm simulator
job = execute(circuit, BasicAer.get_backend('qasm_simulator'), shots=n_shots, memory=True)

measurements = job.result().get_memory()
counts = {0: 0}
for i in range(np.power(2, n_ms)):
    counts[i] = 0
for i in measurements:
    counts[int(i[2:], 2)] += 1

fig = plot_histogram(counts)

ax = fig.axes[0]
ax.set_ylim(0, 1)

##############################
# Change histogram name here #
##############################

fig.savefig('error_model_histogram.png')
print("Execution completed! Histogram saved.")
