import sys
import os

project_root = os.path.abspath("..")
sys.path.append(project_root)

import qiskit
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from pauliopt.circuits import Circuit
from copy import deepcopy, copy


# Target device:
from qiskit.providers.fake_provider import FakeBoeblingenV2
from pauliopt.topologies import Topology



from qiskit import transpile

# Add all gates to PauliPolynomial in reverse
from pauliopt.pauli.pauli_polynomial import PauliPolynomial, PauliGadget
from pauliopt.clifford.tableau import CliffordTableau, CliffordGate
from pauliopt.pauli.pauli_gadget import PPhase, I, X, Z
from pauliopt.utils import Angle, pi


def synthesize_pp_and_ct(pauliopt_circuit, num_qubits):
    # Create CliffordTableau and PauliPolynomial
    ct = CliffordTableau(num_qubits)
    pp = PauliPolynomial(num_qubits)
    # Fill the PauliPolynomial and CliffordTableau with the circuit
    from itertools import product
    for gate in reversed(pauliopt_circuit.gates):
        if isinstance(gate, CliffordGate):
            pp.propagate_inplace(gate)
            ct.append_gate(gate)
        elif gate.name == "CCX":
            # Add that
            for paulistring in product([I, Z], [I,Z], [I,X]): 
                if paulistring == (I, I, I):
                    continue
                angle = Angle(1 / 4) if paulistring.count(X) % 2 == 1 else Angle(-1 / 4)
                final_paulistring = [I if i not in gate.qubits else paulistring[gate.qubits.index(i)] for i in range(num_qubits)]
                pp >>= PPhase(angle) @ final_paulistring
        elif gate.name == "T":
            final_paulistring = [I]*num_qubits
            final_paulistring[gate.qubits[0]] = Z
            pp >>= PPhase(Angle(1/8)) @ final_paulistring
        else:
            raise NotImplementedError(f"Gate {gate.name} not implemented yet in the script.")
    return pp, ct
# Optimize PauliPolynomial

def synthesize_qiskit_alternative(pp: PauliPolynomial, ct: CliffordTableau) -> QuantumCircuit:
    circuit = QuantumCircuit(pp.num_qubits)
    from qiskit.quantum_info import SparsePauliOp, Clifford
    import numpy as np
    def parse_gadget(gadget: PauliGadget) -> str:
        return "".join([str(l)[-1] for l in gadget.paulis])
    
    def parse_angle(gadget:PauliGadget) -> float:
        return gadget.angle
    strings = [parse_gadget(gadget) for gadget in pp.pauli_gadgets]
    angles = [parse_angle(g) for g in pp.pauli_gadgets]
    print(strings[:4], angles[:4])
    qiskit_pp = SparsePauliOp(strings, angles)
    print(ct.tableau.shape, ct.signs.shape)
    numpy_ct = np.zeros((2*pp.num_qubits, 2*pp.num_qubits+1))
    numpy_ct[:,:-1] = ct.tableau
    numpy_ct[:,-1] = ct.signs
    qiskit_ct = Clifford(numpy_ct)
    circuit.compose(qiskit_pp, inplace=True, qubits=range(0, pp.num_qubits), clbits=[])
    circuit.compose(qiskit_ct, inplace=True, qubits=range(0, pp.num_qubits), clbits=[])
    return circuit

def print_pp_info(pp: PauliPolynomial, label: str):
    print(f"{label} PauliPolynomial info:")
    print(f"  Number of terms: {len(pp.pauli_gadgets)}")
    #num_nontrivial = sum(1 for term in pp.pauli_gadgets if not term.is_identity())
    #print(f"  Number of non-trivial terms: {num_nontrivial}")
    print(f"  Number of T gates (pi/4 phases): {sum(1 for term in pp.pauli_gadgets if term.angle.value*4 % 2 != 0)}")
    print()

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
    return naive_circuit

def lex_synth(pp, ct: CliffordTableau, topo):
    # Synthesize PauliPolynomial
    lex_pp_pauliopt, _, trailing_clifford = pauli_polynomial_steiner_gray_clifford(deepcopy(pp), copy(topo), return_tableau=True)
    # Make into Qiskit circuit
    lex_pp = lex_pp_pauliopt.to_qiskit()
    print("Lex pp circuit:", lex_pp.count_ops())
    # Prepend the trailing cliffords
    for gate in trailing_clifford.gates:
        ct.append_gate(gate)
    # Synthesize the new CliffordTableau
    lex_ct_pauliopt = synthesize_tableau_perm_row_col(ct, topo=copy(topo))
    lex_ct = lex_ct_pauliopt.to_qiskit()
    # Combine the two parts
    lex_circuit = lex_pp.compose(lex_ct)
    #print("Lex ct circuit:", lex_ct.count_ops())
    print("Lex Synthesized circuit:", lex_circuit.count_ops())
    return lex_circuit

    # ----------------------------------------------#

def factoring21(qiskit_circuit):
    # math 
    from qiskit import Aer, execute
    from fractions import Fraction
    import math 
    import pandas as pd
    import matplotlib.pyplot as plt

    shotcount = 1000
    sim = Aer.get_backend('qasm_simulator')
    job = execute(qiskit_circuit, sim, shots=shotcount)
    result = job.result()
    count = result.get_counts(qiskit_circuit)
    print(count)
    # -----------------------------------------------------------------------------------------
    # organize data 

    number_hit =  [] 
    frequency  =  []
    fraction   =  []
    denominator = []

    for measured_value in count:
        num = measured_value[::].replace(' ', '')  # remove spaces
        value = count.get(measured_value[::])
        frequency.append(value)
        number_hit.append(int(num, 2))  
        frac = Fraction(int(num, 2),1024).limit_denominator(21)
        fraction.append(frac)
        denominator.append(frac.denominator)
    
    plt.bar(number_hit, frequency)
    plt.show()
    

    df = pd.DataFrame ({
        "number hit": number_hit,
        "frequency": frequency,
        "fraction": fraction,
        "denominator": denominator
    })

    print("Original data sampled from the quantum circuit: ")
    print(df)
    filtered_df = df[df['frequency'] > shotcount*0.04] # equivalent to more than 40 counts for every 1000 shots  

    # print("Remove some noisy element that have low count: ")
    # print(filtered_df)

    period = []
    for num in filtered_df['denominator']:
        if num not in period and num % 2 == 0:
            period.append(num)

    print("possible period value to find the factors: ")
    print(period)

    for num in period: 
        guess1 = 2**(num//2) + 1 
        guess2 = 2**(num//2) - 1
        if math.gcd(guess1, 21) not in [1, 21]:
            # print("from period =", num)
            print("factor found: ", math.gcd(guess1, 21))
        if math.gcd(guess2, 21) not in [1, 21]:
            print("factor found: ", math.gcd(guess2, 21))

def split_circuit_before_measurement(qiskit_circuit):
    nqubits = qiskit_circuit.num_qubits
    nbits = qiskit_circuit.num_clbits
    new_circuit = QuantumCircuit(nqubits, nbits)
    new_circuit.compose(qiskit_circuit, inplace=True)
    circuit = new_circuit.copy_empty_like()
    post_circuit = new_circuit.copy_empty_like()
    barrier_index = -1
    for i, instr in enumerate(new_circuit.data):
        if instr.operation.name == "barrier":
            barrier_index = i
            break
    for instr in new_circuit.data[:barrier_index]:
        circuit.append(instr.operation, instr.qubits, instr.clbits)
    print("Circuit without final measurements", circuit.count_ops())
    
    for instr in new_circuit.data[barrier_index:]:
        post_circuit.append(instr.operation, instr.qubits, instr.clbits)
    print("Post barrier circuit", post_circuit.count_ops())
    return circuit, post_circuit

def generate_permutation_circuit(qiskit_circuit):
    d = qiskit_circuit.count_ops()
    key = next(k for k in d if k.startswith('permutation'))
    pattern = key.split("_",1) [1].strip("[]")
    permutation_table = [int(x) for x in pattern.split(",") if x]
    from qiskit.circuit.library import Permutation
    permutation_circuit = Permutation(num_qubits=qiskit_circuit.num_qubits, pattern=permutation_table)
    return permutation_circuit

def generate_reverse_permutation_circuit(qiskit_circuit):
    permutation_circuit = generate_permutation_circuit(qiskit_circuit)
    permutation_circuit_inverse = permutation_circuit.inverse()
    return permutation_circuit_inverse

if __name__ == "__main__":
    do_routing = False
    true_QFT = True
    
    if do_routing:
        backend = FakeBoeblingenV2()
        edges = backend.coupling_map.get_edges()
        topo = Topology(backend.num_qubits, {e for e in backend.coupling_map.get_edges()})
    else:
        backend = None
        topo = Topology.complete(15)

    if true_QFT:
        qasm_name = "factor21_properQFT.qasm"
    else:
        qasm_name = "factor21.qasm"

    #qasm to qiskit    
    qiskit_circuit = QuantumCircuit.from_qasm_file(qasm_name)
    new_circuit, post_circuit = split_circuit_before_measurement(qiskit_circuit)
    num_qubits = new_circuit.num_qubits
    use_debug_circuit = False
    if use_debug_circuit:
        new_circuit = QuantumCircuit(num_qubits)
        new_circuit.h(0)
        new_circuit.s(0)
        new_circuit.h(0)
        new_circuit.cx(0,1)
        new_circuit.t(1)
        new_circuit.cx(0,1)
        new_circuit.h(0)
        new_circuit.sdg(0)
        new_circuit.h(0)
        post_circuit = new_circuit.copy_empty_like()

    qiskit_transpile = transpile(new_circuit, backend=backend, basis_gates=['h', 'cx', 'x', 'rz', 'sx', 's', 'sxdg'], optimization_level=3)
    print("Qiskit transpiled circuit:", qiskit_transpile.count_ops())
    
    #qiskit to pauliopt
    pauliopt_circuit = None 
    pauliopt_circuit = Circuit.from_qiskit(new_circuit)
    pp, ct = synthesize_pp_and_ct(pauliopt_circuit, num_qubits)

    check_qiskit_alternative = False
    if check_qiskit_alternative:
        qiskit_alt = transpile(synthesize_qiskit_alternative(pp, ct), backend=backend, basis_gates=['h', 'cx', 'x', 'rz', 'sx', 's', 'sxdg'], optimization_level=3)

        from qiskit.quantum_info import Statevector

        original_op = Statevector(new_circuit)
        qiskit_op = Statevector(qiskit_transpile)
        qiskit_alt_op = Statevector(qiskit_alt)
        print("Normal transpile", original_op.equiv(qiskit_op))
        print("Alt qiskit", original_op.equiv(qiskit_alt_op))
        exit(-1)
    
    from pauliopt.pauli.simplification.simple_simplify import simplify_pauli_polynomial

    pp = simplify_pauli_polynomial(pp, allow_acs=False)
    print_pp_info(pp, "Optimized")
    alt_pp = simplify_pauli_polynomial(pp, allow_acs=True)
    print_pp_info(alt_pp, "Shuffled = Wrong")
    #pauliopt to qiskit 
    naive_circuit_no_acs = naive_synth(deepcopy(pp), deepcopy(ct), topo)
    naive_circuit_acs = naive_synth(deepcopy(alt_pp), deepcopy(ct), topo)
    lex_circuit_no_acs = lex_synth(deepcopy(pp), deepcopy(ct), topo)
    lex_circuit_acs = lex_synth(deepcopy(alt_pp), deepcopy(ct), topo)
    print(lex_circuit_no_acs)

    # print(lex_circuit_acs)
    permutation = generate_permutation_circuit(lex_circuit_no_acs)
    reversed_permutation = generate_reverse_permutation_circuit(lex_circuit_no_acs)

    
    
    del lex_circuit_no_acs.data[5676]
    print("Circuit after permutation deletion: ", lex_circuit_no_acs.count_ops())
    
    if use_debug_circuit:
        print("Original circuit:", new_circuit, sep="\n")
        print("Naive circuit no acs:", naive_circuit_no_acs, sep="\n")
        print("Naive circuit acs:", naive_circuit_acs, sep="\n")
        print("Lex circuit no acs:", lex_circuit_no_acs, sep="\n")
        print("Lex circuit acs:", lex_circuit_acs, sep="\n")  
 
    check_statevector = False
    if check_statevector:
        from qiskit.quantum_info import Statevector

        original_op = Statevector(new_circuit)
        qiskit_op = Statevector(qiskit_transpile)
        naive_acs_op = Statevector(naive_circuit_acs)
        naive_acs_with_perm_op = Statevector(naive_circuit_acs.compose(generate_permutation_circuit(lex_circuit_acs)))
        naive_circuit_no_acs_op = Statevector(naive_circuit_no_acs)
        naive_no_acs_with_perm_op = Statevector(naive_circuit_no_acs.compose(generate_permutation_circuit(lex_circuit_no_acs)))
        lex_op_acs = Statevector(lex_circuit_acs)
        lex_op_no_acs = Statevector(lex_circuit_no_acs)

        print("qiskit", original_op.equiv(qiskit_op))
        print("naive acs", original_op.equiv(naive_acs_op))
        print("naive no acs", original_op.equiv(naive_circuit_no_acs_op))
        print("lex acs", original_op.equiv(lex_op_acs))
        print("lex no acs", original_op.equiv(lex_op_no_acs))

        print("Are we internally consistent?", naive_circuit_no_acs_op.equiv(lex_op_no_acs))
        print("And with acs?", naive_acs_op.equiv(lex_op_acs))
        # assert(original_op.equiv(naive_circuit_no_acs_op))



    #add permutation and measurement after barrier back 
    qiskit_transpile_full = qiskit_transpile.compose(post_circuit)
    naive_circuit_no_acs_full = naive_circuit_no_acs.compose(post_circuit)
    naive_circuit_acs_full = naive_circuit_acs.compose(post_circuit)

    lex_circuit_no_acs_full = lex_circuit_no_acs.compose(permutation).compose(reversed_permutation).compose(post_circuit)
    
    # run and makesure factoring 21 is correct
    print("\n\n\n")
    print("original circuit result")
    factoring21(qiskit_circuit)
    # print("qiskit transpiled result")
    # factoring21(qiskit_transpile_full)
    # print("naive_circuit_no_acs_result")
    # factoring21(naive_circuit_no_acs_full)
    # print("naive_circuit_acs_result")
    # factoring21(naive_circuit_acs_full)
    print("lex_circuit_no_acs_result")
    factoring21(lex_circuit_no_acs_full)
    # print("lex_circuit_acs_result")
    # factoring21(lex_circuit_acs_full)

