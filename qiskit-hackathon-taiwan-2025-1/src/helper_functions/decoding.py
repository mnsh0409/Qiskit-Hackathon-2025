from typing import List, Tuple  # For typing annotation.
from qiskit.circuit import QuantumCircuit, Parameter # Importing QuantumCircuit for circuit operations.
import torch  # Importing torch for tensor operations.
import numpy as np

def decode_actions_into_circuit(
    action: List[int], 
    current_circuit: QuantumCircuit, 
    current_params: List[float],
    num_qubits: int
) -> Tuple[bool, List[float]]:
    """
    Decode agent actions into quantum circuit operations and update parameters.

    Args:
        action (List[int]): Agent action [gate_type, target_qubit, control_qubit/angle_index]
        current_circuit (QuantumCircuit): Current quantum circuit
        current_params (List[float]): Current list of parameters
        num_qubits (int): Number of qubits in the circuit

    Returns:
        Tuple[bool, List[float]]: (valid_action, new_params)
    """
    # Define gate types and their properties
    gate_types = ['h', 'x', 'y', 'z', 'cx', 'cz', 'rx', 'ry', 'rz', 'rzz', 't', 'sx']
    
    # Unpack action components
    gate_type_idx = action[0]
    target_qubit = action[1]
    control_or_angle_idx = action[2]
    
    # Validate qubit indices
    if target_qubit >= num_qubits:
        return False, current_params
    
    # Get gate name
    try:
        gate_name = gate_types[gate_type_idx]
    except IndexError:
        return False, current_params
    
    # Initialize new parameters list
    new_params = current_params.copy()
    valid_action = True
    
    try:
        # Handle two-qubit gates
        if gate_name in ['cx', 'cz']:
            if control_or_angle_idx >= num_qubits or control_or_angle_idx == target_qubit:
                return False, current_params
                
            if gate_name == 'cx':
                current_circuit.cx(control_or_angle_idx, target_qubit)
            else:  # 'cz'
                current_circuit.cz(control_or_angle_idx, target_qubit)
        
        # Handle parameterized gates
        elif gate_name in ['rx', 'ry', 'rz']:
            # Normalize angle index to [0, 2π]
            angle = (control_or_angle_idx / 63) * 2 * np.pi
            
            # Add new parameter
            param = Parameter(f'θ_{len(new_params)}')
            new_params.append(angle)
            
            # Apply parameterized gate
            if gate_name == 'rx':
                current_circuit.rx(param, target_qubit)
            elif gate_name == 'ry':
                current_circuit.ry(param, target_qubit)
            else:  # 'rz'
                current_circuit.rz(param, target_qubit)
        
        # Handle single-qubit non-parameterized gates
        else:
            if gate_name == 'h':
                current_circuit.h(target_qubit)
            elif gate_name == 'x':
                current_circuit.x(target_qubit)
            elif gate_name == 'y':
                current_circuit.y(target_qubit)
            elif gate_name == 'z':
                current_circuit.z(target_qubit)
            elif gate_name == 't':
                current_circuit.t(target_qubit)
            elif gate_name == 's':
                current_circuit.s(target_qubit)
            elif gate_name == 'sx':
                current_circuit.sx(target_qubit)
    
    except Exception as e:
        print(f"Error applying gate: {e}")
        valid_action = False
    
    return valid_action, new_params
    #pass