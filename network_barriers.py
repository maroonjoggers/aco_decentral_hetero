import numpy as np
from cvxopt import matrix, solvers
import utils

def network_barriers(U, X, env):
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

    #Define agents to ignore this for
    returning_agents_indices = []
    for agent in env.agents:
        if agent.state == "Returning":
            returning_agents_indices.append(agent.id)

    N = X.shape[1]

    P = matrix(2*np.eye(2*N))
    q = matrix(-2*np.reshape(U, 2*N, order='F'))

    A = np.zeros((N, 2*N))
    b = np.zeros(N)

    for i, row in enumerate(A):           #Each row corresponds to an agent (and its closest neighbor)

        #Find closest neighbor
        #Define distances
        pos_i = X[:,i].reshape(2,1)
        diffs = X - pos_i
        dists = np.linalg.norm(diffs, axis=0)
        dists[i] = np.inf                       #ignore ourself

        #ignore those which are returning
        for idx in returning_agents_indices:
            dists[idx] = np.inf

        j = np.argmin(dists)

        ACTIVATION_THRESHOLD = radii[i] / 2

        if dists[j] <= radii[i] and i not in returning_agents_indices and dists[j] >= ACTIVATION_THRESHOLD:
            h = radii[i]**2 - dists[j]**2                              #TODO: Not sure if radii list is actually how we'll access things
            b[i] = -GAMMA*h**3
            
            #populate A matrix
            A[i, i*2] = 2*(X[0,i]-X[0,j])
            A[i, i*2 + 1] = 2*(X[1,i]-X[1,j])
            A[i, j*2] = -2*(X[0,i]-X[0,j])
            A[i, j*2 + 1] = -2*(X[1,i]-X[1,j])
        else:
            pass

    A = matrix(A)
    b = matrix(b)

    qpSolution = solvers.qp(P,q,A,b)
    newInput = np.array(qpSolution['x'])

    newU = np.reshape(newInput, (2, -1), order="F")

    return newU

