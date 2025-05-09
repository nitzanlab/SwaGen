# SWAGEN: Enhancing Swarms' Durability via Graph Signal Processing and GNN-based Generative Modeling

## Overview

SWAGEN is a framework designed to enhance the durability of swarms—such as animal groups or artificial drone formations—against external perturbations, particularly predation. The method leverages graph signal processing (GSP) and graph neural networks (GNNs) to analyze and optimize swarm configurations for improved resilience.

## Key Contributions

* **Graph-Based Swarm Analysis**: Utilizes GSP to model swarm configurations and assess their response to external threats.
* **Generative Modeling with GNN**: Introduces a GraphSAGE-based generative model that optimizes swarm structures by balancing detectability and diffusion properties.
* **Novel Configurations**: Discovers optimized spatial configurations that simultaneously minimize predation risks and enhance survivability.


## Repository Structure

```
├── swagen/                      # Datasets for swarm configurations
├── models/                    # SWAGEN generative models
├── scripts/                   # Training and evaluation scripts
├── visualization/             # Scripts and notebooks for visualizations
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Installation

Clone this repository:

```bash
git clone https://github.com/nitzanlab/SwaGen.git
cd SwaGen
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Downloading the Code

The core codebase files for SWAGEN are available below. Download each file and place them in the `scripts/` folder of the repository:

* [train.py](#)
* [shapes.py](#)
* [models.py](#)
* [losses.py](#)
* [data\_utils.py](#)
* [simulation.py](#)

Ensure the files maintain their names exactly as provided.

## Usage

### Training

To train the SWAGEN model, run:

```bash
python scripts/train.py
```

### Evaluating

To evaluate and visualize swarm durability:

```bash
python scripts/simulation.py
```

### Generating Configurations

Generate optimized swarm configurations:

```bash
python scripts/train.py --generate
```

## Dependencies

* PyTorch
* PyTorch Geometric
* NumPy
* Matplotlib

Complete dependencies can be found in [requirements.txt](requirements.txt).

## Citation

If you use this work, please cite:

```
@article{karin2024swagen,
  title={Enhancing Swarms' Durability to Threats via Graph Signal Processing and GNN-based Generative Modeling},
  author={Jonathan Karin, Zoe Piran, Mor Nitzan},
  year={2024}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions or collaboration, please contact [Mor Nitzan](mailto:mor.nitzan@mail.huji.ac.il).
