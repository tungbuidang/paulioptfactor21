import sys 
import os 

project_root = os.path.abspath("..")
sys.path.append(project_root)

import qiskit
from qiskit.circuit.library import SwapGate
from qiskit import QuantumCircuit

qc = QuantumCircuit(3)
qc.h(0)
qc.h(1)
qc.h(2)
qc.swap(0,1)
qc.swap(1,2)
qc.swap(0,2)
print("Original circuit:")
print(qc.draw())

# use pauliopt on qc 
from pauliopt.circuits import Circuit
from pauliopt.pauli.pauli_polynomial import PauliPolynomial, PauliGadget
from pauliopt.clifford.tableau import CliffordTableau, CliffordGate
from pauliopt.pauli.pauli_gadget import PPhase, I, X, Z
from pauliopt.utils import Angle, pi
from itertools import product 
from pauliopt.topologies import Topology

pauliopt_circuit = Circuit.from_qiskit(qc)
pp = PauliPolynomial(3)
ct = CliffordTableau(3)
for gate in reversed(pauliopt_circuit.gates):
    if isinstance(gate, CliffordGate):
        pp.propagate_inplace(gate)
        ct.prepend_gate(gate)
    elif gate.name == "CCX":
        # Add that
        for paulistring in product([I, Z], [I,Z], [I,X]): 
            if paulistring == (I, I, I):
                continue
            angle = Angle(1 / 4) if paulistring.count(X) % 2 == 1 else Angle(-1 / 4)
            final_paulistring = [I if i not in gate.qubits else paulistring[gate.qubits.index(i)] for i in range(num_qubits)]
            pp >>= PPhase(angle) @ final_paulistring
    else:
        raise NotImplementedError(f"Gate {gate.name} not implemented yet in the script.")

topo = Topology.complete(3)

from pauliopt.pauli.synthesis.steiner_gray_synthesis import pauli_polynomial_steiner_gray_clifford
from pauliopt.clifford.tableau_synthesis import synthesize_tableau_perm_row_col
from copy import copy

lex_pp_pauliopt, _, trailing_clifford = pauli_polynomial_steiner_gray_clifford(pp, copy(topo), return_tableau=True)
# Make into Qiskit circuit
lex_pp = lex_pp_pauliopt.to_qiskit()
print("Lex pp circuit:", lex_pp.count_ops())
# Prepend the trailing cliffords
for gate in trailing_clifford.gates:
    ct.prepend_gate(gate)
# Synthesize the new CliffordTableau
lex_ct_pauliopt = synthesize_tableau_perm_row_col(ct, topo=copy(topo))
lex_ct = lex_ct_pauliopt.to_qiskit()
# Combine the two parts
lex_circuit = lex_pp.compose(lex_ct)
#print("Lex ct circuit:", lex_ct.count_ops())
print("Lex Synthesized circuit:", lex_circuit.count_ops())
print(lex_circuit)