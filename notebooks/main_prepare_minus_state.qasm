// Generated by Classiq.
// Classiq version: 0.36.1
// Creation timestamp: 2024-02-03T18:56:37.990587+00:00
// Random seed: 2096435800

OPENQASM 2.0;
include "qelib1.inc";
gate main_prepare_minus_state q0 {
  x q0;
  h q0;
}

qreg q[1];
main_prepare_minus_state q[0];