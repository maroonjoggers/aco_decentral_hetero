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
    GAMMA = 200.0

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


# Not used
def network_barriers_with_lambda_v0(U, X, env, lambda_values):
    '''
    Ensures input keeps at least 1 agent within connection (as long as we started within connection),
    and solves the full lambda-weighted optimization:
    
    Minimize:
        lambda * ||u - u_pheromone||^2 + (1 - lambda) * t
    where:
        t >= u^T * u_pheromone
        t >= - u^T * u_pheromone

    Inputs:
        U: Agent velocities SI (2xN) (u_pheromone vector!)
        X: Agent states SI (2xN)
        env: Environment object (for agent states)
        lambda_value: RL agent output for this agent
    Outputs:
        newU: New agent velocities (2xN), optimized
    '''

    GAMMA = 20.0

    radii = utils.communication_radius_list()

    # Define agents to ignore (those returning)
    returning_agents_indices = []
    for agent in env.agents:
        if agent.state == "Returning":
            returning_agents_indices.append(agent.id)

    N = X.shape[1] 
    total_vars = 3 * N  # u (2N) + auxiliary variable t (N)

    # Construct P (Quadratic term)
    P = np.zeros((total_vars, total_vars))
    for i in range(N):
        P[2*i : 2*(i+1), 2*i : 2*(i+1)] = 2 * lambda_values[i] * np.eye(2)
    P = matrix(P)

    # Construct q (Linear term)
    u_pheromone_flat = np.reshape(U, 2 * N, order='F')
    q = np.zeros((total_vars, ))
    for i in range(N):
        u_ph_i_flat = U[:, i]  # (2,)
        q[2 * i : 2 * (i + 1)] = - (1 + lambda_values[i]) * u_ph_i_flat
        q[2 * N + i] = + (1 - lambda_values[i])
    q = matrix(q)

    # Build original connectivity constraints A u <= b
    A_list = []
    b_list = []

    for i in range(N):
        # Find closest neighbor
        pos_i = X[:, i].reshape(2, 1)
        diffs = X - pos_i
        dists = np.linalg.norm(diffs, axis=0)
        dists[i] = np.inf  # Ignore self

        # Ignore returning agents
        for idx in returning_agents_indices:
            dists[idx] = np.inf

        j = np.argmin(dists)

        if dists[j] <= radii[i] and i not in returning_agents_indices:
            h = radii[i] ** 2 - dists[j] ** 2
            b_list.append(-GAMMA * h ** 3)

            A_row = np.zeros((total_vars,))
            # Populate A matrix entries
            A_row[i * 3] = 2 * (X[0, i] - X[0, j])
            A_row[i * 3 + 1] = 2 * (X[1, i] - X[1, j])
            A_row[j * 3] = -2 * (X[0, i] - X[0, j])
            A_row[j * 3 + 1] = -2 * (X[1, i] - X[1, j])
            # Note: auxiliary variable t has no contribution here
            A_list.append(A_row)
        else:
            pass 

    # Add new constraints for auxiliary variable t to model absolute value

    for i in range(N):
        # Constraint 1: u_i^T * u_pheromone_i - t_i <= 0
        A_t1 = np.zeros((total_vars,))
        A_t1[2 * i : 2 * (i + 1)] = U[:, i]
        A_t1[2 * N + i] = -1.0
        A_list.append(A_t1)
        b_list.append(0.0)

        # Constraint 2: -u_i^T * u_pheromone_i - t_i <= 0
        A_t2 = np.zeros((total_vars,))
        A_t2[2 * i : 2 * (i + 1)] = -U[:, i]
        A_t2[2 * N + i] = -1.0
        A_list.append(A_t2)
        b_list.append(0.0)

    # Stack A and b
    A = matrix(np.vstack(A_list))
    b = matrix(np.array(b_list))

    # Solve QP
    solvers.options['show_progress'] = False  # Optional: silence solver output
    qpSolution = solvers.qp(P, q, A, b)
    newInput = np.array(qpSolution['x']).flatten()

    # Extract u variables (ignore auxiliary variable t)
    u_star = newInput[:2 * N]
    newU = np.reshape(u_star, (2, -1), order="F")

    # Optional debug logging
    print(f"[QP] Lambda: {lambda_values}")
    print(f"[QP] u_pheromone_flat: {u_pheromone_flat}")
    print(f"[QP] Solution u*: {u_star}")
    print(f"[QP] Auxiliary variables t*: {newInput[2 * N: 2 * N + N]}")

    return newU


def network_barriers_with_lambda(U, X, env, lambda_values):
    '''
    Ensures input keeps at least 1 agent within connection (as long as we started within connection),
    and solves the full lambda-weighted optimization:

    Minimize:
        lambda * ||u - u_pheromone||^2 + (1 - lambda) * (u^T u_pheromone)^2

    Inputs:
        U: Agent velocities SI (2xN) (u_pheromone vector)
        X: Agent states SI (2xN)
        env: Environment object (for agent states)
        lambda_values: RL agent output lambda per agent (array of N floats)

    Outputs:
        newU: New agent velocities (2xN), optimized
    '''

    GAMMA = 200.0

    radii = utils.communication_radius_list()

    # Define agents to ignore (those returning)
    returning_agents_indices = []
    for agent in env.agents:
        if agent.state == "Returning":
            returning_agents_indices.append(agent.id)

    N = X.shape[1] 
    total_vars = 2 * N  # Only u variables 

    # === Build P matrix (quadratic term) ===
    P = np.zeros((total_vars, total_vars))

    for i in range(N):
        lambda_i = lambda_values[i]
        u_ph_i = U[:, i].reshape(2, 1)  # (2, 1)

        # Term 1: lambda_i * (u - u_ph_i)^2 = lambda_i * (u^T u) - 2 * lambda_i * (u^T u_ph_i) + constant
        P[2*i : 2*(i+1), 2*i : 2*(i+1)] += 2 * lambda_i * np.eye(2)

        # Term 2: (1 - lambda_i) * (u^T u_ph_i)^2
        # This expands to u^T (u_ph_i u_ph_i^T) u
        Q_i = np.outer(u_ph_i, u_ph_i)  # (2x2)
        P[2*i : 2*(i+1), 2*i : 2*(i+1)] += 2 * (1 - lambda_i) * Q_i

    P = matrix(P)

    # === Build q vector (linear term) ===
    q = np.zeros((total_vars, ))
    for i in range(N):
        lambda_i = lambda_values[i]
        u_ph_i = U[:, i]

        # Term from expanding (u - u_ph_i)^2
        q[2 * i : 2 * (i + 1)] = -2 * lambda_i * u_ph_i

    q = matrix(q)

    '''
    # === Build A and b (connectivity constraints) ===
    A_list = []
    b_list = []

    for i in range(N):
        # Find closest neighbor
        pos_i = X[:, i].reshape(2, 1)
        diffs = X - pos_i
        dists = np.linalg.norm(diffs, axis=0)
        dists[i] = np.inf  # Ignore self

        # Ignore returning agents
        for idx in returning_agents_indices:
            dists[idx] = np.inf

        j = np.argmin(dists)

        ACTIVATION_THRESHOLD = radii[i] / 2

        if dists[j] <= radii[i] and i not in returning_agents_indices and dists[j] >= ACTIVATION_THRESHOLD:
            h = radii[i] ** 2 - dists[j] ** 2
            b_list.append(-GAMMA * h ** 3)

            A_row = np.zeros((total_vars,))
            # Populate A matrix entries
            A_row[i * 2] = 2 * (X[0, i] - X[0, j])
            A_row[i * 2 + 1] = 2 * (X[1, i] - X[1, j])
            A_row[j * 2] = -2 * (X[0, i] - X[0, j])
            A_row[j * 2 + 1] = -2 * (X[1, i] - X[1, j])

            A_list.append(A_row)

    if A_list:
        A = matrix(np.vstack(A_list))
        b = matrix(np.array(b_list))
    else:
        # No connectivity constraints, use empty matrix
        A = matrix(np.zeros((1, total_vars)))
        b = matrix(np.zeros(1))
    '''

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

    # === Solve QP ===
    solvers.options['show_progress'] = False  # Optional: silence solver output
    qpSolution = solvers.qp(P, q, A, b)

    newInput = np.array(qpSolution['x']).flatten()

    newU = np.reshape(newInput, (2, -1), order="F")

    # Optional debug logging
    print(f"[QP] Lambda: {lambda_values}")
    print(f"[QP] Solution u*: {newU}")

    return newU