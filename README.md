# Qiskit-Hackathlon-2025
Quantum Reinforcement Learning for VQE

This project uses a Proximal Policy Optimization (PPO) agent to train a quantum circuit for finding the ground state energy of a molecule (LiH) using the Variational Quantum Eigensolver (VQE) algorithm.

## 🚀 Getting Started

### Prerequisites

-   Python 3.9+
-   Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/qiskit-hackathon-2025.git](https://github.com/YOUR_USERNAME/qiskit-hackathon-2025.git)
    cd qiskit-hackathon-2025
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Setting Up the Development Environment

[Download](https://www.anaconda.com/download) and install Anaconda on Linux:

```bash
chmod +x <filename>.sh && ./<filename>.sh
```

### PyTorch Setup

- Create and activate the environment:

```bash
conda env create -f torch_env.yml && conda activate torch-qiskit-gpu
```

- Verify GPU:

```bash
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

### TensorFlow Setup

- Create and activate the environment:

```bash
conda env create -f tf_gpu_env.yml && conda activate tf-qiskit-gpu
```

- Verify GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Running the Code

To start the training process, run the `main.py` script with the configuration file as a command-line argument:

```bash
python src/main.py src/config_lih.cfg
