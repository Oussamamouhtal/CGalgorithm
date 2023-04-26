"""
  Module: Iterative Solvers 
  Author: Oussama MOUHTAL
"""
import numpy as np
import time


def CG(A, b, tol, maxit, ortho = True):
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
    flag = 0                      # initialize a flag variable to check if the algorithm converged
    r = b                         # set the initial residual to b
    p = b                         # set the initial search direction to b
    rs_old = np.dot(r, r)         # compute the initial squared residual
    x = np.zeros_like(b)          # initialize the current approximation to a zero vector
    nrmb = np.linalg.norm(b)
    for i in range(maxit):  
        if ortho:
            if i == 0:
                R = np.array([r/(rs_old**0.5)]).T
            else:
                R = np.append(R, np.array([r/(rs_old**0.5)]).T, axis=1)
        q = A.dot(p)          
        alpha = rs_old / np.dot(p, q)   # compute the step size
        x = x + alpha * p               # update the current approximation x
        r = r - alpha * q               # update the residual
        # Re orthogonalisation
        if ortho:
            r = r - np.dot(R,np.dot(R.T,r))
        rs_new = np.dot(r, r)           # compute the new squared residual
        print('iteration : {:<9d} residues : {:<20.4f}  || xstar - x || : {:<9.3f} || x || {: 9.9f}'.format(i, np.sqrt(rs_new), np.linalg.norm(x - xstar), np.linalg.norm(x)))

        if np.sqrt(rs_new)  < tol * nrmb:     
            flag = 1                    # set the flag variable to indicate convergence
            end_time = time.time()      # record the end time
            print(flag)
            break 
        p = r + (rs_new / rs_old) * p  # update the search direction
        rs_old = rs_new                # update the old squared residual
    if flag == 0: 
        end_time = time.time()   
        print("Maximum number of iterations reached without convergence, and the time taken {}".format(end_time - start_time))
    return x, np.sqrt(rs_new) / nrmb, i, flag 







if __name__ == '__main__':
    n = 5000
    A = np.random.rand(n,n)
    A = np.matmul(A, A.T)
    b = np.random.rand(n)
    xstar = np.linalg.solve(A, b)    # calculate the exact solution
    maxit = 5000
    tol   = 1e-8
    print(" > Solution with Conjugate Gradient Algorithm")
    cg_sol, residus, iter, flag = CG(A, b, tol, maxit)
    print("|| xstar - cg_sol || : ", np.linalg.norm(xstar-cg_sol),'\n')  

