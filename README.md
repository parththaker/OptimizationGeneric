# Optmization : Simulation Benchmark

*NOTE: UNDER CONSTRUCTION*

This project includes different types of cost functions, dataset and other supporting
functionality to help new optimization algorithms run different type of simulations environment.

## Requirment
* Python 2.7

## Cost functions
* Logistic regression
* Linear regression
* Differentiable cost function
* Non-differentiable cost function

## Graphs models
* Complete graph
* Line graph
* Star graph
* Erdos - Renyi graph

## Optimization methods
* DLADMM
* DADMM
* Vanilla gradient descent

## Dataset
* Binary model generated {-1, 1}
* Linear model generated

## Usage

### root.py 

Basic module to run update algorithms. 

#### Options 
 --function_type : Choice of cost function
 
 --method_type : Choice of update algorithm
 
 --err_lim : The acceptable error bound
 
 --step_size : Choice of stepsize for the update algorithm
 
 --plot : Flag to include if plotting of the run is required
 
 --stoc : Flag to include if function and gradient values are required to have stochasticity.
 
 #### Example run
 
 ```python
python root.py --function_type monkey --method_type gradtent --err_lim 0.001 --step_size 0.1 --plot
```
 
## Authors

* **[Parth Thaker](https://parththaker.github.io/)**

