"""
  Module: Iterative Solvers 
  Author: Oussama MOUHTAL
"""
import numpy as np
import time


def CG(A, b, tol, maxit):
    """ 
    Conjugate Gradient Algorithm
        CG solves the symmetric positive definite linear system 
            A x  = b 
        A     : n-by-n symmetric and positive definite matrix
        b     : n dimensional right hand side vector
        maxit : maximum number of iterations
        tol   : error tolerance on the residual 
    """
    start_time = time.time()      # start timing the execution
    xstar = np.linalg.solve(A, b)   # compute the exact solution
    flag = 0                      # initialize a flag variable to check if the algorithm converged
    r = b                         # set the initial residual to b
    p = b                         # set the initial search direction to b
    rs_old = np.dot(r, r)         # compute the initial squared residual
    x = np.zeros_like(b)          # initialize the current approximation to a zero vector
    for i in range(maxit):  
        q = np.dot(A, p)          
        alpha = rs_old / np.dot(p, q)   # compute the step size
        x = x + alpha * p               # update the current approximation x
        r = r - alpha * q               # update the residual
        rs_new = np.dot(r, r)           # compute the new squared residual
        if np.sqrt(rs_new)  < tol:     
            flag = 1                    # set the flag variable to indicate convergence
            end_time = time.time()      # record the end time
            break 
        p = r + (rs_new / rs_old) * p  # update the search direction
        rs_old = rs_new                # update the old squared residual
        if i % 5 == 0:                 # print the current iteration number, squared residual, error, and norm every 5 iterations
            print('iteration : {:<9d} residues : {:<20.4f}  || xstar - x || : {:<9.3f} || xstar || {: 9.9f}'.format(i, np.sqrt(rs_new), np.linalg.norm(x - xstar), np.linalg.norm(x)))
    if flag == 0: 
        end_time = time.time()   
        print("Maximum number of iterations reached without convergence, and the time taken {}".format(end_time - start_time))
    return x






if __name__ == '__main__':
    n = 10000
    A = np.random.rand(n,n)
    A = np.matmul(A, A.T)
    P = np.eye(n)
    b = np.random.rand(n)
    x0 = np.zeros_like(b)
    xstar = np.linalg.solve(A, b)
    maxit = 40000
    tol   = 1e-6
    print(" > Solution with Conjugate Gradient Algorithm")
    cg_sol = CG(A, b, tol, maxit)
    print("|| xstar - cg_sol || : ", np.linalg.norm(xstar-cg_sol),'\n')  

