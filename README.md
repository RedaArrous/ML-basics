# Machine Learning Fundamentals

A comprehensive collection of machine learning implementations and experiments, documenting practical applications of core ML algorithms and concepts.

## Overview

This repository contains Jupyter Notebooks and Python implementations covering fundamental machine learning algorithms, from supervised learning techniques to unsupervised clustering methods. Each implementation includes detailed explanations and is designed to demonstrate practical applications of ML concepts.

## Project Structure

```
ML/
├── algorithms/              # Core algorithm utilities and plotting functions
├── Supervised algorithms/   # Supervised learning implementations
│   ├── Breast_cancer.ipynb
│   ├── Linear_regression.ipynb
│   ├── Logistic_regression.ipynb
│   └── Multiple_linear_regression.ipynb
├── Unsupervised algorithms/ # Unsupervised learning implementations
│   └── K-means.ipynb
└── requirements.txt         # Project dependencies
```

## Algorithms Implemented

### Supervised Learning
- **Linear Regression** - Simple and multiple linear regression models
- **Logistic Regression** - Binary classification algorithm
- **Breast Cancer Classification** - Real-world application using classification techniques

### Unsupervised Learning
- **K-Means Clustering** - Partition-based clustering algorithm

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RedaArrous/ML-basics.git
cd ML
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Launch Jupyter Notebook to explore the implementations:
```bash
jupyter notebook
```

Navigate to the desired notebook in either the `Supervised algorithms/` or `Unsupervised algorithms/` directory.

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

See [requirements.txt](requirements.txt) for complete dependencies.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available for educational purposes.

## Contact

For questions or feedback, please open an issue in the repository.
