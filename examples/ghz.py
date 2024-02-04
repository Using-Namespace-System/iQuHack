# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import cirq
import qsimcirq


parser = argparse.ArgumentParser(description='GHZ circuit')
parser.add_argument('--nqubits', type=int, default=3, help='the number of qubits in the circuit')
parser.add_argument('--nsamples', type=int, default=3, help='the number of samples to take')
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


def make_ghz_circuit(nqubits, measure=False):
    qubits = cirq.LineQubit.range(nqubits)
    circuit = cirq.Circuit()
    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.CNOT(qubits[idx], qubits[idx + 1]) for idx in range(nqubits - 1))
    if measure:
        circuit.append(cirq.measure(*qubits))
    return circuit


def main(nqubits=28, nrepetitions=10, ngpus=1):
    measure = True if nrepetitions > 0 else False
    circuit = make_ghz_circuit(nqubits, measure=measure)

    qsim_options = qsim_options_from_arguments(ngpus)
    simulator = qsimcirq.QSimSimulator(qsim_options=qsim_options)
    if nrepetitions > 0:
        results = simulator.run(circuit, repetitions=nrepetitions)
    else:
        results = simulator.simulate(circuit)
    print(results)


if __name__ == '__main__':
    args = parser.parse_args()
    main(nqubits=args.nqubits, nrepetitions=args.nsamples, ngpus=args.ngpus)
