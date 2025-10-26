# GenForge: Sculpting Symbolic Solutions with Multi-Population Genetic Programming

![GenForge Logo](GenForge_Logo.png)  <!-- Optional: Add a logo or banner image -->

## Overview

**GenForge** is an open-source Python package for interpretable symbolic modeling through *Genetic Programming (GP)*.  
It introduces a cohesive ecosystem of modules — `gpclassifier`, `gpregressor`, and `spfp` — that together support the evolution of compact, human-readable models for both classification and regression tasks.  
The framework integrates multi-population evolution, semantic partitioning, and ensemble-based symbolic learning to produce transparent, high-performance models.

### Key Features

- **Multi-Population Genetic Programming**  
  Parallel populations evolve diverse solutions that collectively reduce variance and prevent premature convergence.

- **Symbolic Regression and Classification**  
  - `gpregressor`: for continuous-valued prediction using multi-gene symbolic regression.  
  - `gpclassifier`: for categorical prediction with softmax-calibrated symbolic classifiers.

- **Semantic-Preserving Feature Partitioning (SPFP)**  
  The optional `spfp` module partitions high-dimensional feature spaces into complementary, information-preserving “views.”  
  Each view trains a separate GP population; the resulting submodels are then aggregated to yield multi-view ensembles with stronger generalization and interpretability.

- **Parsimony and Calibration**  
  GenForge emphasizes compact symbolic expressions and calibrated confidence estimates for trustable decision boundaries.

- **Reproducibility and Auditability**  
  Built with fixed random seeds, detailed logs, and exportable equations for transparent evaluation and regulatory use.

- **Educational and Research Utility**  
  Modular design and readable structure make it suitable for both academic teaching and applied research in symbolic machine learning, explainable AI, and scientific modeling.

## Authors

- **Mohammad Sadegh Khorshidi**  
  Faculty of Information and IT, University of Technology Sydney  
  Email: msadegh.khorshidi.ak@gmail.com  

- **Navid Yazdanjue**  
  Faculty of Information and IT, University of Technology Sydney  
  Email: navid.yazdanjue@gmail.com

- **Hassan Gharoun**  
  Faculty of Information and IT, University of Technology Sydney  
  Email: hassan.gharoun@student.uts.edu.au

- **Mohammad Reza Nikoo**  
  Department of Civil and Architectural Engineering, Sultan Qaboos University, Muscat, Oman  
  Email: m.reza@squ.edu.om  

- **Fang Chen**  
  Faculty of Information and IT, University of Technology Sydney  
  Email: fang.chen@uts.edu.au  

- **Amir H. Gandomi**  
  Faculty of Information and IT, University of Technology Sydney;  
  University Research and Innovation Center (EKIK), Óbuda University  
  Email: gandomi@uts.edu.au  


## Documentation

A detailed *User Manual* describing installation, configuration, and examples is available here:

 [Download the GenForge User Manual (PDF)](./GenForge_UserManual.pdf)

The manual includes:
- Step-by-step setup and environment configuration
- Example scripts for `gpclassifier`, `gpregressor`, and `SPFPPartitioner`
- Guidance for extending and visualizing GenForge runs

### Citation
If you use GenForge in your research, please cite the corresponding papers listed in the [CITATION.cff](CITATION.cff) file or click “Cite this repository” on GitHub.

### Installation

To install GenForge from source:

```bash
git clone https://github.com/maisamkhorshidi/genforge.git
cd genforge
pip install .

