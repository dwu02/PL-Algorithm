#!/usr/bin/python2.7
# Homework 0 Code
import numpy as np
import matplotlib.pyplot as plt
import random
np.seterr(divide = 'ignore') 

def generate_input(N,d):
    temp = np.array([[np.random.uniform(0,1) for i in range(N)]]).T
    for i in range(d-1):
        vector = np.array([[2*np.random.uniform(0,1)-1 for i in range(N)]]).T
        temp = np.concatenate((temp, vector), axis=1)
    return temp

def perceptron_learn(data_in):
    # Run PLA on the input data
    #
    # Inputs: data_in: Assumed to be a matrix with each row representing an
    #                (x,y) pair, with the x vector augmented with an
    #                initial 1 (i.e., x_0), and the label (y) in the last column
    # Outputs: w: A weight vector (should linearly separate the data if it is linearly separable)
    #        iterations: The number of iterations the algorithm ran for

    # Your code here, assign the proper values to w and iterations:
    x,y,w = data_in[:, [i for i in range(1,data_in.shape[1]-1)]],data_in[:, [data_in.shape[1]-1]],np.array([[0 for i in range(data_in.shape[1]-2)]]).T
    iterations,condition = 0,False
    while condition == False:
        y_hat = np.sign(np.dot(x[0:data_in.shape[0],],w[0:]))
        diff_y = y-y_hat
        if np.linalg.norm(diff_y) != 0: 
            iterations+=1
            w = w+(np.dot(x[0:data_in.shape[0],].T,diff_y[:]))
        else: condition = True 
    return w, iterations


def perceptron_experiment(N, d, num_exp):
    # Code for running the perceptron experiment in HW0
    # Implement the dataset construction and call perceptron_learn; repeat num_exp times
    #
    # Inputs: N is the number of training data points
    #         d is the dimensionality of each data point (before adding x_0)
    #         num_exp is the number of times to repeat the experiment
    # Outputs: num_iters is the # of iterations PLA takes for each experiment
    #          bounds_minus_ni is the difference between the theoretical bound and the actual number of iterations
    # (both the outputs should be num_exp long)

    # Initialize the return variables
    num_iters = np.zeros((num_exp,))
    bounds_minus_ni = np.zeros((num_exp,))

    # Your code here, assign the values to num_iters and bounds_minus_ni:
    for i in range(1,num_exp):
        x = generate_input(N,d)
        y = np.sign(np.dot(x[0:N,],np.array([[0]+[np.random.uniform(0,1) for j in range(d)]]).T[1:]))
        w,iteration = perceptron_learn(np.concatenate((np.array([[1 for k in range(N)]]).T, x,y), axis=1))
        num_iters[i] = iteration
        p,r,w_norm = np.amin(y*(w.T*x))*-1,np.amax(np.apply_along_axis(np.linalg.norm, 1, x)),np.linalg.norm(w)
        t = ((r*r)*(w_norm*w_norm))/(p*p)
        bounds_minus_ni[i] = t-iteration
    bounds_minus_ni = bounds_minus_ni[bounds_minus_ni != 0]
    return num_iters, bounds_minus_ni

def main():
    print("Running the experiment...")
    num_iters, bounds_minus_ni = perceptron_experiment(100, 10, 1000)

    print("Printing histogram...")
    plt.hist(num_iters)
    plt.title("Histogram of Number of Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Count")
    plt.show()

    print("Printing second histogram")
    log_bounds_minus_ni = np.log(bounds_minus_ni)
    log_bounds_minus_ni = log_bounds_minus_ni[~np.isnan(log_bounds_minus_ni)]
    plt.hist(log_bounds_minus_ni)
    plt.title("Bounds Minus Iterations")
    plt.xlabel("Log Difference of Theoretical Bounds and Actual # Iterations")
    plt.ylabel("Count")
    plt.show()

if __name__ == "__main__":
    main()
