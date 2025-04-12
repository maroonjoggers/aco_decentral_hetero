import numpy as np
from cvxopt import matrix, solvers
import utils

def network_barriers(U, X):
    '''
    Ensures input keeps at least 1 agent within connection (as long as we started within connection)

    Inputs:
        U: Agent velocities SI (2xN)
        X: Agent states SI (2xN)
        radii: List which has the communication radius for robot i
    Outputs:
        newU: New agent velocity
    '''
    GAMMA = 20.0

    radii = utils.communication_radius_list()

    N = X.shape[1]

    P = matrix(2*np.eye(2*N))
    q = matrix(-2*np.reshape(U, 2*N, order='F'))

    A = np.zeros((N, 2*N))
    b = np.zeros(N)

    for i, row in enumerate(A):           #Each row corresponds to an agent (and its closest neighbor)

        #Find closest neighbor
        pos_i = X[:,i].reshape(2,1)
        diffs = X - pos_i
        dists = np.linalg.norm(diffs, axis=0)
        dists[i] = np.inf
        j = np.argmin(dists)


        h = radii[i]**2 - dists[j]**2                              #TODO: Not sure if radii list is actually how we'll access things
        b[i] = -GAMMA*h**3
        
        #populate
        A[i, i*2] = 2*(X[0,i]-X[0,j])
        A[i, i*2 + 1] = 2*(X[1,i]-X[1,j])
        A[i, j*2] = -2*(X[0,i]-X[0,j])
        A[i, j*2 + 1] = -2*(X[1,i]-X[1,j])

    A = matrix(A)
    b = matrix(b)

    qpSolution = solvers.qp(P,q,A,b)
    newInput = np.array(qpSolution['x'])

    newU = np.reshape(newInput, (2, -1), order="F")

    return newU

