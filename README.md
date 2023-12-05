# Learn to Rapidly and Robustly Optimize Hybrid Precoding

Python repository for the paper:

O. Lavi and N. Shlezinger, "Learn to Rapidly and Robustly Optimize Hybrid Precoding," in IEEE Transactions on Communications, vol. 71, no. 10, pp. 5814-5830, Oct. 2023. 

https://ieeexplore.ieee.org/document/10173560

Please cite our paper if the code is used for publishing research.

## Introduction
This repository implements the proposed classical projected gradient ascent (PGA) algorithm and the unfolded PGA.

Our suggested method involves using deep learning tools in order to learn the hyperparameters of an iterative optimizer (PGA), for the purpose of making the optimizer achieve the best possible performance on the training data in a fixed and low number of iterations. The learning of these hyperparameters is regarded as an offline step, requiring only some representative channel realization that are used as the training data. 

In the learning procedure, the initial step sizes are set to be the constant value for which the PGA converged, as in that case the algorithm starts from a solid initialization that is further improved by leveraging data.Upon completion of the learning procedure, the values of the step-sizes are retained, and are utilized by the model-based optimizer operating with this predetermined number of iterations. 

The algorithm with the learned step-sizes values and the fixed number of iterations is what we refer to as the unfolded algorithm. 

## Python Code
### Description
The code includes one file named 'main.py' in which the DNN model is being defined and the training procedure is being executed as well.

### Execution
To execute the code you can use any Python IDE, we used PyCharm.

You need to install the following Python libraries:
* NumPy
* PyTorch
* Matplotlib

