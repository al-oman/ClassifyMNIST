# MNIST Digit Classifier

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://docs.python.org/3/whatsnew/3.13.html)
[![Numpy](https://img.shields.io/badge/numpy-2.2.2-blue.svg)](https://github.com/numpy/numpy/releases/tag/v2.2.2)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

A neural network based digit classifier made without the use of ML frameworks such as PyTorch, Tensorflow, etc. This is designed to be used as a test bench for understanding the innerworkings of ML techniques. 

## Table of Contents:
- [Capabilities](#capabilities)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Capabilities

Current:
- Stochastic gradient descent (SGD)
- Layer normalization
- ReLU activation functions
- Softmax at output layer

Planned: 
- Adam optimizer
- Minibatching
- Learning rate scheduling

## Installation
1. Clone the repo:
```bash 
git clone https://github.com/al-oman/ClassifyMNIST.git
```
2. Start using!

## Usage
- uncomment the train() and/or run() functions
- adjust the parameters and run the file

```bash
if __name__ == "__main__":
    #run('nets/net99.pkl')
    #train(low_res=True, alpha=1e-2, batch_size=1)
```
## License

MIT License.