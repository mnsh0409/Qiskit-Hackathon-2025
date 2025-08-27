from typing import List  # For typing annotation.
from qiskit.circuit import QuantumCircuit, Gate, Parameter  # Importing QuantumCircuit for circuit operations.
import torch  # Importing torch for tensor operations.
import numpy as np

def encode_circuit_into_input_embedding(qc: QuantumCircuit) -> np.ndarray:
    """
    Encode the quantum circuit into a tensor representation with features for:
    - Gate type (one-hot encoded)
    - Target qubit index (normalized)
    - Control qubit index (if applicable, normalized)
    - Rotation angle (if applicable, normalized to [0, 1])
    - Gate sequence position (normalized)
    - Connectivity flag (0 or 1 indicating if control qubit is used)
    
    Args:
        qc (QuantumCircuit): The quantum circuit to be encoded.

    Returns:
        torch.Tensor: The tensor representation of the quantum circuit.
    """
    # Supported gate types and their indices
    gate_types = ['h', 'x', 'y', 'z', 'cx', 'cz', 'rx', 'ry', 'rz', 'rzz', 't', 'sx']
    num_gate_types = len(gate_types)
    
    # Maximum circuit depth from configuration (LiH: 50)
    max_depth = 50
    
    # Features per gate: 
    # [gate_type_one_hot, target_qubit, control_qubit, angle, position, connectivity_flag]
    num_features = num_gate_types + 5  # 12 + 5 = 17 features
    
    # Initialize tensor with zeros (padded to max_depth)
    embedding = torch.zeros((max_depth, num_features))
    
    # Normalization factors
    num_qubits = qc.num_qubits
    divisor = max(1, num_qubits - 1)  # Avoid division by zero
    
    for idx, instruction in enumerate(qc.data):
        if idx >= max_depth:
            break
            
        gate = instruction.operation
        qubits = [qubit._index for qubit in instruction.qubits]
        
        # Get base gate name (handle parameterized gates)
        base_name = gate.name.lower()
        
        # Encode gate type
        try:
            gate_idx = gate_types.index(base_name)
            embedding[idx, gate_idx] = 1.0
        except ValueError:
            # Unknown gate type - skip encoding
            continue
        
        # Encode target qubit (always the first qubit)
        target_qubit = qubits[0]
        embedding[idx, num_gate_types] = target_qubit / divisor
        
        # Encode control qubit and connectivity (for two-qubit gates)
        if len(qubits) > 1:
            control_qubit = qubits[1]
            embedding[idx, num_gate_types + 1] = control_qubit / divisor
            embedding[idx, num_gate_types + 3] = 1.0  # Connectivity flag
        
        # Encode rotation angle for parameterized gates
        if base_name in ['rx', 'ry', 'rz'] and gate.params:
            angle = gate.params[0]

            # Check if the angle is a symbolic Parameter or a concrete number
            if isinstance(angle, Parameter):
                # If it's a symbolic Parameter, we can't do math on it.
                # Use a default value (e.g., 0.0) for the encoding.
                normalized_angle = 0.0
            else:
                # If it's a concrete number, normalize it as before.
                normalized_angle = (float(angle) % (2 * np.pi)) / (2 * np.pi)

            embedding[idx, num_gate_types + 2] = normalized_angle

        # Encode sequence position
        embedding[idx, num_gate_types + 4] = idx / max_depth
    
    # Flatten the embedding to 1D tensor
    return embedding.flatten().numpy()