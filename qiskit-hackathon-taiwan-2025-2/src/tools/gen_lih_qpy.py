# tools/gen_lih_npz.py
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import JordanWignerMapper
import numpy as np
import os

# LiH 설정 (당신의 cfg와 동일)
mol = MoleculeInfo(
    symbols=["Li","H"],
    coords=([0.0,0.0,0.0],[0.0,0.0,1.571274961436279]),
    multiplicity=1,
    charge=0
)

driver = PySCFDriver.from_molecule(mol, basis="sto3g")
problem = driver.run()

# Active space (6 spatial orbitals, 4 electrons -> (2,2))
problem = ActiveSpaceTransformer(
    num_electrons=4, num_spatial_orbitals=6
).transform(problem)

ham = problem.second_q_ops()[0]
mapper = JordanWignerMapper()
qubit_op = mapper.map(ham)  # SparsePauliOp

# SparsePauliOp -> 리스트(label, complex coeff)
pairs = qubit_op.to_list()  # [(str, complex), ...]
labels = np.array([p[0] for p in pairs], dtype=object)
creal  = np.array([p[1].real for p in pairs], dtype=np.float64)
cimag  = np.array([p[1].imag for p in pairs], dtype=np.float64)
num_qubits = int(qubit_op.num_qubits)

out_path = "src/operators/qubit_op_LiH.npz"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.savez(out_path, labels=labels, creal=creal, cimag=cimag, num_qubits=num_qubits)
print("Wrote:", out_path, "| terms:", len(labels), "| num_qubits:", num_qubits)
