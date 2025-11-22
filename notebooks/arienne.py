import sys
import os

project_root = os.path.abspath("..")
sys.path.append(project_root)

import qiskit
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from pauliopt.circuits import Circuit
from copy import deepcopy, copy

# print(sys.executable)
print("qiskit version is: ")
print(qiskit.__version__)
# print(pauliopt)

# Target device:
from qiskit.providers.fake_provider import FakeBoeblingenV2
from pauliopt.topologies import Topology
do_routing = False

if do_routing:
    backend = FakeBoeblingenV2()
    edges = backend.coupling_map.get_edges()
    topo = Topology(backend.num_qubits, {e for e in backend.coupling_map.get_edges()})
else:
    backend = None
    topo = Topology.complete(15)

#qiskit
original_circuit = QuantumCircuit.from_qasm_file("factor21.qasm")
nqubits = original_circuit.num_qubits
nbits = original_circuit.num_clbits
new_circuit = QuantumCircuit(nqubits, nbits)
new_circuit.compose(original_circuit, inplace=True)
print("Circuit with final measurements:", new_circuit.count_ops())
circuit = new_circuit.copy_empty_like()
barrier_index = -1
for i, instr in enumerate(new_circuit.data):
    if instr.operation.name == "barrier":
        barrier_index = i
        break
for instr in new_circuit.data[:barrier_index]:
    circuit.append(instr.operation, instr.qubits, instr.clbits)
print("Circuit without final measurements", circuit.count_ops())

# Qiskit transpiler
from qiskit import transpile
transpiled_circuit = transpile(circuit, backend=backend, basis_gates=['h', 'cx', 'x', 'rz', 'sx', 's', 'sxdg'], optimization_level=3)
print("Qiskit transpiled circuit:", transpiled_circuit.count_ops())

#pauliopt
pauliopt_circuit = None
pauliopt_circuit = Circuit.from_qiskit(circuit)

# Add all gates to PauliPolynomial in reverse
from pauliopt.pauli.pauli_polynomial import PauliPolynomial, PauliGadget
from pauliopt.clifford.tableau import CliffordTableau, CliffordGate
from pauliopt.pauli.pauli_gadget import PPhase, I, X, Z
from pauliopt.utils import Angle, pi
# Create CliffordTableau and PauliPolynomial
ct = CliffordTableau(circuit.num_qubits)
pp = PauliPolynomial(circuit.num_qubits)
# Fill the PauliPolynomial and CliffordTableau with the circuit
from itertools import product
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
            final_paulistring = [I if i not in gate.qubits else paulistring[gate.qubits.index(i)] for i in range(circuit.num_qubits)]
            pp >>= PPhase(angle) @ final_paulistring
    else:
        raise NotImplementedError(f"Gate {gate.name} not implemented yet in the script.")

# Optimize PauliPolynomial

from pauliopt.pauli.simplification.simple_simplify import simplify_pauli_polynomial
pp = simplify_pauli_polynomial(pp, allow_acs=False)
alt_pp  = simplify_pauli_polynomial(pp, allow_acs=True)

def print_pp_info(pp: PauliPolynomial, label: str):
    print(f"{label} PauliPolynomial info:")
    print(f"  Number of terms: {len(pp.pauli_gadgets)}")
    #num_nontrivial = sum(1 for term in pp.pauli_gadgets if not term.is_identity())
    #print(f"  Number of non-trivial terms: {num_nontrivial}")
    print(f"  Number of T gates (pi/4 phases): {sum(1 for term in pp.pauli_gadgets if term.angle.value*4 % 2 != 0)}")
    print()

print_pp_info(pp, "Original")
print_pp_info(alt_pp, "Shuffled = Wrong")

# Synthesize back to circuit
from pauliopt.clifford.tableau_synthesis import synthesize_tableau, synthesize_tableau_perm_row_col
from pauliopt.pauli.synthesis.steiner_gray_synthesis import pauli_polynomial_steiner_gray_clifford
def naive_synth(pp, ct, topo):
    naive_pp = pp.to_qiskit(topology=copy(topo))
    naive_ct_pauliopt, perm = synthesize_tableau(ct, topo=copy(topo))
    naive_ct = naive_ct_pauliopt.to_qiskit()
    naive_circuit = naive_pp.compose(naive_ct)
    #print("Naive pp circuit:", naive_pp.count_ops())
    #print("Naive ct circuit:", naive_ct.count_ops())
    print("Naive Synthesized circuit:", naive_circuit.count_ops())

#naive_synth(deepcopy(pp), deepcopy(ct)), topo)
#naive_synth(deepcopy(alt_pp), deepcopy(ct), topo)

def lex_synth(pp, ct: CliffordTableau, topo):
    # Synthesize PauliPolynomial
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

lex_synth(deepcopy(pp), deepcopy(ct), topo)
lex_synth(deepcopy(alt_pp), deepcopy(ct), topo)