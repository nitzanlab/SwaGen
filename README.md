# SWAGEN: Enhancing Swarms' Durability via Graph Signal Processing and GNN-based Generative Modeling

## Overview

SWAGEN is a framework designed to enhance the durability of swarms—such as animal groups or artificial drone formations—against external perturbations, particularly predation. The method leverages graph signal processing (GSP) and graph neural networks (GNNs) to analyze and optimize swarm configurations for improved resilience.

## Key Contributions

* **Graph-Based Swarm Analysis**: Utilizes GSP to model swarm configurations and assess their response to external threats.
* **Generative Modeling with GNN**: Introduces a GraphSAGE-based generative model that optimizes swarm structures by balancing detectability and diffusion properties.
* **Novel Configurations**: Discovers optimized spatial configurations that simultaneously minimize predation risks and enhance survivability.


## Reproducement of the 'kite'
For easy reproducement of the 'kite' please see the followign notebook:
https://github.com/nitzanlab/SwaGen/blob/main/reproducing_kite.ipynb

## Installation

Clone this repository:

```bash
git clone https://github.com/nitzanlab/SwaGen.git
cd SwaGen
```

Install:

```bash
pip install .
```

## Downloading the Code

The core codebase files for SWAGEN are available below. Download each file and place them in the `swagen/` folder of the repository:

* [train.py](#)
* [shapes.py](#)
* [models.py](#)
* [losses.py](#)
* [data\_utils.py](#)
* [simulation.py](#)

Ensure the files maintain their names exactly as provided.

## Usage

## Generating swarm shapes

Use the shape functions in `shapes.py` to create structured 2D swarm formations:

```python
from shapes import (
    arrange_agents_in_rectangle,
    arrange_agents_in_arrow,
    arrange_agents_in_v,
    arrange_agents_in_v_triangle_wave,
    arrange_agents_in_kite
)

points = arrange_agents_in_rectangle(N=100, alpha=5)
```

### Training

To train the SWAGEN model, run:

```bash
python scripts/train.py
```

### Evaluating

To run the predation simulation ( replace 'generate_agent_positions' with the swarm shape):

```bash
python scripts/simulation.py
```
## Dependencies

* PyTorch
* PyTorch Geometric
* NumPy
* Matplotlib

## Citation

If you use this work, please cite:

```
@article{karin2025enhancing,
  title={Enhancing Swarms Durability to Threats via Graph Signal Processing and GNN-based Generative Modeling},
  author={Karin, Jonathan and Piran, Zoe and Nitzan, Mor},
  journal={arXiv preprint arXiv:2507.03039},
  year={2025}
}
```
## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions or collaboration, please contact [Jonathan Karin](mailto:jonathan.karin@mail.huji.ac.il).
