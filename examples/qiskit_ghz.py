# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from mpi4py import MPI
import argparse


def create_ghz_circuit(n_qubits):
    circuit = QuantumCircuit(n_qubits)
    circuit.h(0)
    for qubit in range(n_qubits - 1):
        circuit.cx(qubit, qubit + 1)
    return circuit


def run(n_qubits, precision, use_cusvaer):
    simulator = Aer.get_backend('aer_simulator_statevector')
    simulator.set_option('cusvaer_enable', use_cusvaer)
    simulator.set_option('precision', precision)
    circuit = create_ghz_circuit(n_qubits)
    circuit.measure_all()
    circuit = transpile(circuit, simulator)
    job = simulator.run(circuit)
    result = job.result()

    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f'precision: {precision}')
        print(result.get_counts())
        print(f'backend: {result.backend_name}')


parser = argparse.ArgumentParser(description="Qiskit ghz.")
parser.add_argument('--nbits', type=int, default=20, help='the number of qubits')
parser.add_argument('--precision', type=str, default='single', choices=['single', 'double'], help='numerical precision')
parser.add_argument('--disable-cusvaer', default=False, action='store_true', help='disable cusvaer')

args = parser.parse_args()

run(args.nbits, args.precision, not args.disable_cusvaer)
