# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from collections import Counter
import argparse
import numpy as np
import scipy as sp
import cirq
import qsimcirq

"""Demonstrates Simon's algorithm.
Simon's Algorithm solves the following problem:

Given a function  f:{0,1}^n -> {0,1}^n, such that for some s ∈ {0,1}^n,

f(x) = f(y) iff  x ⨁ y ∈ {0^n, s},

find the n-bit string s.

A classical algorithm requires O(2^n/2) queries to find s, while Simon’s
algorithm needs only O(n) quantum queries.

=== REFERENCE ===
D. R. Simon. On the power of quantum cryptography. In35th FOCS, pages 116–123,
Santa Fe,New Mexico, 1994. IEEE Computer Society Press.

=== EXAMPLE OUTPUT ===
Secret string = [1, 0, 0, 1, 0, 0]
Circuit:
                ┌──────┐   ┌───────────┐
(0, 0): ────H────@──────────@─────@──────H───M('result')───
                 │          │     │          │
(1, 0): ────H────┼@─────────┼─────┼──────H───M─────────────
                 ││         │     │          │
(2, 0): ────H────┼┼@────────┼─────┼──────H───M─────────────
                 │││        │     │          │
(3, 0): ────H────┼┼┼@───────┼─────┼──────H───M─────────────
                 ││││       │     │          │
(4, 0): ────H────┼┼┼┼@──────┼─────┼──────H───M─────────────
                 │││││      │     │          │
(5, 0): ────H────┼┼┼┼┼@─────┼─────┼──────H───M─────────────
                 ││││││     │     │
(6, 0): ─────────X┼┼┼┼┼─────X─────┼───×────────────────────
                  │││││           │   │
(7, 0): ──────────X┼┼┼┼───────────┼───┼────────────────────
                   ││││           │   │
(8, 0): ───────────X┼┼┼───────────┼───┼────────────────────
                    │││           │   │
(9, 0): ────────────X┼┼───────────X───×────────────────────
                     ││
(10, 0): ────────────X┼────────────────────────────────────
                      │
(11, 0): ─────────────X────────────────────────────────────
                └──────┘   └───────────┘
Most common Simon Algorithm answer is: ('[1 0 0 1 0 0]', 100)

***If the input string is s=0^n, no significant answer can be
distinguished (since the null-space of the system of equations
provided by the measurements gives a random vector). This will
lead to low frequency count in output string.
"""


parser = argparse.ArgumentParser(description="Simon's algorithm.")
parser.add_argument('--nbits', type=int, default=3, help='the number of bits in the secret string')
parser.add_argument('--ngpus', type=int, default=1, help='the number of GPUs to use')


def create_qsim_options(
    max_fused_gate_size=2,
    disable_gpu=False,
    cpu_threads=1,
    gpu_mode=(0,),
    verbosity=0,
    n_subsvs=-1,
    use_sampler=None,
    debug=False
):
    return qsimcirq.QSimOptions(
        max_fused_gate_size=max_fused_gate_size,
        disable_gpu=disable_gpu,
        cpu_threads=cpu_threads,
        gpu_mode=gpu_mode,
        verbosity=verbosity,
        n_subsvs=n_subsvs,
        use_sampler=use_sampler,
        debug=debug
    )


def qsim_options_from_arguments(ngpus):
    if ngpus > 1:
        return create_qsim_options(gpu_mode=ngpus)
    elif ngpus == 1:
        return create_qsim_options()
    elif ngpus == 0:
        return create_qsim_options(disable_gpu=True, gpu_mode=0, use_sampler=False)

def main(qubit_count=3, ngpus=1):

    data = []  # we'll store here the results

    # define a secret string:
    secret_string = np.random.randint(2, size=qubit_count)

    print(f'Secret string = {secret_string}')

    if ngpus > 0:
        qsim_options = qsim_options_from_arguments(ngpus)
        simulator = qsimcirq.QSimSimulator(qsim_options=qsim_options)
    else:
        simulator = cirq.Simulator()

    n_samples = 100
    for _ in range(n_samples):
        flag = False  # check if we have a linearly independent set of measures
        while not flag:
            # Choose qubits to use.
            input_qubits = [cirq.GridQubit(i, 0) for i in range(qubit_count)]  # input x
            output_qubits = [
                cirq.GridQubit(i + qubit_count, 0) for i in range(qubit_count)
            ]  # output f(x)

            # Pick coefficients for the oracle and create a circuit to query it.
            oracle = make_oracle(input_qubits, output_qubits, secret_string)

            # Embed oracle into special quantum circuit querying it exactly once
            circuit = make_simon_circuit(input_qubits, output_qubits, oracle)

            # Sample from the circuit n-1 times (n = qubit_count).
            results = simulator.run(circuit, repetitions=qubit_count-1).measurements['result']

            # Classical Post-Processing:
            flag = post_processing(data, results)

    freqs = Counter(data)
    print('Circuit:')
    print(circuit)
    print(f'Most common answer was : {freqs.most_common(1)[0]}')


def make_oracle(input_qubits, output_qubits, secret_string):
    """Gates implementing the function f(a) = f(b) iff a ⨁ b = s"""
    # Copy contents to output qubits:
    for control_qubit, target_qubit in zip(input_qubits, output_qubits):
        yield cirq.CNOT(control_qubit, target_qubit)

    # Create mapping:
    if sum(secret_string):  # check if the secret string is non-zero
        # Find significant bit of secret string (first non-zero bit)
        significant = list(secret_string).index(1)

        # Add secret string to input according to the significant bit:
        for j in range(len(secret_string)):
            if secret_string[j] > 0:
                yield cirq.CNOT(input_qubits[significant], output_qubits[j])
    # Apply a random permutation:
    pos = [
        0,
        len(secret_string) - 1,
    ]  # Swap some qubits to define oracle. We choose first and last:
    yield cirq.SWAP(output_qubits[pos[0]], output_qubits[pos[1]])


def make_simon_circuit(input_qubits, output_qubits, oracle):
    """Solves for the secret period s of a 2-to-1 function such that
    f(x) = f(y) iff x ⨁ y = s
    """

    c = cirq.Circuit()

    # Initialize qubits.
    c.append(
        [
            cirq.H.on_each(*input_qubits),
        ]
    )

    # Query oracle.
    c.append(oracle)

    # Measure in X basis.
    c.append([cirq.H.on_each(*input_qubits), cirq.measure(*input_qubits, key='result')])

    return c


def post_processing(data, results):
    """Solves a system of equations with modulo 2 numbers"""
    sing_values = sp.linalg.svdvals(results)
    tolerance = 1e-5
    if sum(sing_values < tolerance) == 0:  # check if measurements are linearly dependent
        flag = True
        null_space = sp.linalg.null_space(results).T[0]
        solution = np.around(null_space, 3)  # chop very small values
        minval = abs(min(solution[np.nonzero(solution)], key=abs))
        solution = (solution / minval % 2).astype(int)  # renormalize vector mod 2
        data.append(str(solution))
        return flag


if __name__ == '__main__':
    args = parser.parse_args()
    main(qubit_count=args.nbits, ngpus=args.ngpus)
