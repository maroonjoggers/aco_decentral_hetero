import numpy as np
from cvxopt import matrix, solvers
import utils
from scipy.special import comb
from cvxopt.blas import dot

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

    # Threshold control inputs before QP
    magnitude_limit = 0.2
    norms = np.linalg.norm(U, 2, 0)
    idxs_to_normalize = (norms > magnitude_limit)
    U[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

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

    GAMMA = 2000.0

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
    # print(f"[QP] Lambda: {lambda_values}")
    # print(f"[QP] Solution u*: {newU}")

    return newU







def network_barriers_obstacles(U, X, env):
    '''
    Ensures input keeps at least 1 agent within connection (as long as we started within connection)

    Inputs:
        U: Agent velocities SI (2xN)
        X: Agent states SI (2xN)
        radii: List which has the communication radius for robot i
    Outputs:
        newU: New agent velocity
    '''
    GAMMA_safety = 5.0
    GAMMA_network = 200.0

    safety_radius = 0.125
    radii = utils.communication_radius_list()

    #Define agents to ignore this for
    returning_agents_indices = []
    for agent in env.agents:
        if agent.state == "Returning":
            returning_agents_indices.append(agent.id)

    N = X.shape[1]

    P = matrix(2*np.eye(2*N))
    #q = matrix(-2*np.reshape(U, 2*N, order='F'))


    #SAFETY COLLISION BARRIERS

    combinations = int(comb(N, 2))
    A_safety = np.zeros((combinations, 2*N))
    b_safety = np.zeros((combinations))

    i, j = 0, 1
    for row in range(combinations):
        #Assign A & B
        h = (X[0,i]-X[0,j])**2 + (X[1,i]-X[1,j])**2 - safety_radius**2
        b_safety[row] = GAMMA_safety*h**3

        A_safety[row, i*2] = -2*(X[0,i]-X[0,j])
        A_safety[row, i*2 + 1] = -2*(X[1,i]-X[1,j])
        A_safety[row, j*2] = 2*(X[0,i]-X[0,j])
        A_safety[row, j*2 + 1] = 2*(X[1,i]-X[1,j])

        if j < N-1:
            j += 1
        else:
            i += 1
            j = i + 1




    #NETWORK BARRIERS

    A_network = np.zeros((N, 2*N))
    b_network = np.zeros(N)

    for i, row in enumerate(A_network):           #Each row corresponds to an agent (and its closest neighbor)

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
            h = radii[i]**2 - dists[j]**2       
            b_network[i] = -GAMMA_network*h**3
            
            #populate A matrix
            A_network[i, i*2] = 2*(X[0,i]-X[0,j])
            A_network[i, i*2 + 1] = 2*(X[1,i]-X[1,j])
            A_network[i, j*2] = -2*(X[0,i]-X[0,j])
            A_network[i, j*2 + 1] = -2*(X[1,i]-X[1,j])
        else:
            pass



    
    #COMBINE BARRIERS
    A = matrix(np.vstack((A_safety, A_network)))
    b = matrix(np.hstack((b_safety, b_network)))



    # Threshold control inputs before QP
    magnitude_limit = 0.2
    norms = np.linalg.norm(U, 2, 0)
    idxs_to_normalize = (norms > magnitude_limit)
    U[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]


    #OLD QP SOLUTION WHICH BROKE IT SOMEHOW (FORMATTING OF INPUTS WRONG??)
    # qpSolution = solvers.qp(P,q,A,b)
    # newInput = np.array(qpSolution['x'])
    # newU = np.reshape(newInput, (2, -1), order="F")
    # #return newU

    #NEW QP SOLUTION WHICH WORKS
    q = -2*np.reshape(U, (2*N,1), order='F')
    b = np.reshape(b, (len(b),1), order='F')
    result = solvers.qp(matrix(P), matrix(q), matrix(A), matrix(b))['x']

    return np.reshape(result, (2, N), order='F')




def network_barriers_obstacles2(U, X, env):
    safety_radius = 0.17
    barrier_gain = 100
    boundary_points = np.array([-1.6, 1.6, -1.0, 1.0])
    magnitude_limit = 0.2

    # Initialize some variables for computational savings
    N = U.shape[1]
    num_constraints = int(comb(N, 2)) + 4*N
    A = np.zeros((num_constraints, 2*N))
    b = np.zeros(num_constraints)
    #H = sparse(matrix(2*np.identity(2*N)))
    H = 2*np.identity(2*N)

    count = 0
    for i in range(N-1):
        for j in range(i+1, N):
            error = X[:, i] - X[:, j]
            h = (error[0]*error[0] + error[1]*error[1]) - np.power(safety_radius, 2)

            A[count, (2*i, (2*i+1))] = -2*error
            A[count, (2*j, (2*j+1))] = 2*error
            b[count] = barrier_gain*np.power(h, 3)

            count += 1
    
    for k in range(N):
        #Pos Y
        A[count, (2*k, 2*k+1)] = np.array([0,1])
        b[count] = 0.4*barrier_gain*(boundary_points[3] - safety_radius/2 - X[1,k])**3;
        count += 1

        #Neg Y
        A[count, (2*k, 2*k+1)] = -np.array([0,1])
        b[count] = 0.4*barrier_gain*(-boundary_points[2] - safety_radius/2 + X[1,k])**3;
        count += 1

        #Pos X
        A[count, (2*k, 2*k+1)] = np.array([1,0])
        b[count] = 0.4*barrier_gain*(boundary_points[1] - safety_radius/2 - X[0,k])**3;
        count += 1

        #Neg X
        A[count, (2*k, 2*k+1)] = -np.array([1,0])
        b[count] = 0.4*barrier_gain*(-boundary_points[0] - safety_radius/2 + X[0,k])**3;
        count += 1


    
    #NETWORK BARRIERS
    radii = utils.communication_radius_list()
    GAMMA_network = 10.0

    #Define agents to ignore this for
    returning_agents_indices = []
    for agent in env.agents:
        if agent.state == "Returning":
            returning_agents_indices.append(agent.id)

    A_network = np.zeros((N, 2*N))
    b_network = np.zeros(N)

    for i, row in enumerate(A_network):           #Each row corresponds to an agent (and its closest neighbor)

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

        ACTIVATION_THRESHOLD = radii[i] / 1.2

        if dists[j] <= radii[i] and i not in returning_agents_indices and dists[j] >= ACTIVATION_THRESHOLD:
            h = radii[i]**2 - dists[j]**2       
            b_network[i] = -GAMMA_network*h**3
            
            #populate A matrix
            A_network[i, i*2] = 2*(X[0,i]-X[0,j])
            A_network[i, i*2 + 1] = 2*(X[1,i]-X[1,j])
            A_network[i, j*2] = -2*(X[0,i]-X[0,j])
            A_network[i, j*2 + 1] = -2*(X[1,i]-X[1,j])
        else:
            pass


    #COMBINE BARRIERS
    A = matrix(np.vstack((A, A_network)))
    b = matrix(np.hstack((b, b_network)))



    
    # Threshold control inputs before QP
    norms = np.linalg.norm(U, 2, 0)
    idxs_to_normalize = (norms > magnitude_limit)
    U[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

    f = -2*np.reshape(U, (2*N,1), order='F')
    b = np.reshape(b, (len(b),1), order='F')
    result = solvers.qp(matrix(H), matrix(f), matrix(A), matrix(b))['x']
    #result = solver2.solve_qp(H, f, A, b, 0)[0]

    return np.reshape(result, (2, N), order='F')



def network_barriers_with_obstacles(U, X, env, lambda_values, use_lambda):
    safety_radius = 0.17
    barrier_gain = 100
    boundary_points = np.array([-1.6, 1.6, -1.0, 1.0])
    magnitude_limit = 0.2

    # Initialize some variables for computational savings
    N = U.shape[1]
    num_constraints = int(comb(N, 2)) + 4*N
    A = np.zeros((num_constraints, 2*N))
    b = np.zeros(num_constraints)
    H = 2*np.identity(2*N)

    count = 0
    for i in range(N-1):
        for j in range(i+1, N):
            error = X[:, i] - X[:, j]
            h = (error[0]*error[0] + error[1]*error[1]) - np.power(safety_radius, 2)

            A[count, (2*i, (2*i+1))] = -2*error
            A[count, (2*j, (2*j+1))] = 2*error
            b[count] = barrier_gain*np.power(h, 3)

            count += 1
    
    for k in range(N):
        #Pos Y
        A[count, (2*k, 2*k+1)] = np.array([0,1])
        b[count] = 0.4*barrier_gain*(boundary_points[3] - safety_radius/2 - X[1,k])**3;
        count += 1

        #Neg Y
        A[count, (2*k, 2*k+1)] = -np.array([0,1])
        b[count] = 0.4*barrier_gain*(-boundary_points[2] - safety_radius/2 + X[1,k])**3;
        count += 1

        #Pos X
        A[count, (2*k, 2*k+1)] = np.array([1,0])
        b[count] = 0.4*barrier_gain*(boundary_points[1] - safety_radius/2 - X[0,k])**3;
        count += 1

        #Neg X
        A[count, (2*k, 2*k+1)] = -np.array([1,0])
        b[count] = 0.4*barrier_gain*(-boundary_points[0] - safety_radius/2 + X[0,k])**3;
        count += 1


    
    #NETWORK BARRIERS
    radii = utils.communication_radius_list()
    GAMMA_network = 10.0

    #Define agents to ignore this for
    returning_agents_indices = []
    for agent in env.agents:
        if agent.state == "Returning":
            returning_agents_indices.append(agent.id)

    A_network = np.zeros((N, 2*N))
    b_network = np.zeros(N)

    for i, row in enumerate(A_network):           #Each row corresponds to an agent (and its closest neighbor)

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

        ACTIVATION_THRESHOLD = radii[i] / 1.2

        if dists[j] <= radii[i] and i not in returning_agents_indices and dists[j] >= ACTIVATION_THRESHOLD:
            h = radii[i]**2 - dists[j]**2       
            b_network[i] = -GAMMA_network*h**3
            
            #populate A matrix
            A_network[i, i*2] = 2*(X[0,i]-X[0,j])
            A_network[i, i*2 + 1] = 2*(X[1,i]-X[1,j])
            A_network[i, j*2] = -2*(X[0,i]-X[0,j])
            A_network[i, j*2 + 1] = -2*(X[1,i]-X[1,j])
        else:
            pass

    
    #OBSTACLE BARRIERS
    num_obstacles = len(utils.OBSTACLE_LOCATIONS)
    num_obstacle_constraints = N*num_obstacles

    A_ob = np.zeros((num_obstacle_constraints, 2*N))
    b_ob = np.zeros((num_obstacle_constraints))

    #sr = safety_radius
    ob_barrier_gain = barrier_gain

    count = 0
    for i in range(N):
        for j in range(num_obstacles):
            ob_x, ob_y = utils.OBSTACLE_LOCATIONS[j]["center"]
            #sr = max(safety_radius, utils.OBSTACLE_LOCATIONS[j]["radius"])
            sr = (safety_radius + utils.OBSTACLE_LOCATIONS[j]["radius"])/2

            h = (X[0,i] - ob_x)**2 + (X[1,i] - ob_y)**2 - sr**2
            b_ob[count] = ob_barrier_gain*np.power(h, 3)
            A_ob[count, 2*i]   = -2*(X[0,i] - ob_x)
            A_ob[count, 2*i+1] = -2*(X[1,i] - ob_y)
            count += 1


    #COMBINE BARRIERS
    A = matrix(np.vstack((A, A_network, A_ob)))
    b = matrix(np.hstack((b, b_network, b_ob)))



    # LAMBDA STUFF
    if use_lambda:
        total_vars = 2 * N  # Only u variables 

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
    else:
        P = H




    
    # Threshold control inputs before QP
    norms = np.linalg.norm(U, 2, 0)
    idxs_to_normalize = (norms > magnitude_limit)
    U[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

    f = -2*np.reshape(U, (2*N,1), order='F')
    b = np.reshape(b, (len(b),1), order='F')
    result = solvers.qp(matrix(P), matrix(f), matrix(A), matrix(b))['x']            #use H for no lambda, P for lambda

    return np.reshape(result, (2, N), order='F')


def network_barriers_with_obstacles_safe(U, X, env, lambda_values, use_lambda):
    safety_radius = 0.17
    barrier_gain = 100
    boundary_points = np.array([-1.6, 1.6, -1.0, 1.0])
    magnitude_limit = 0.2

    # Initialize some variables for computational savings
    N = U.shape[1]
    num_constraints = int(comb(N, 2)) + 4*N
    A = np.zeros((num_constraints, 2*N))
    b = np.zeros(num_constraints)
    H = 2*np.identity(2*N)

    count = 0
    for i in range(N-1):
        for j in range(i+1, N):
            error = X[:, i] - X[:, j]
            h = (error[0]*error[0] + error[1]*error[1]) - np.power(safety_radius, 2)
            h = max(h, 1e-4)

            A[count, (2*i, 2*i+1)] = -2*error
            A[count, (2*j, 2*j+1)] = 2*error
            b[count] = barrier_gain*np.power(h, 3)
            count += 1

    for k in range(N):
        #Pos Y
        h = boundary_points[3] - safety_radius/2 - X[1,k]
        h = max(h, 1e-4)
        A[count, (2*k, 2*k+1)] = np.array([0,1])
        b[count] = 0.4*barrier_gain*h**3
        count += 1

        #Neg Y
        h = -boundary_points[2] - safety_radius/2 + X[1,k]
        h = max(h, 1e-4)
        A[count, (2*k, 2*k+1)] = -np.array([0,1])
        b[count] = 0.4*barrier_gain*h**3
        count += 1

        #Pos X
        h = boundary_points[1] - safety_radius/2 - X[0,k]
        h = max(h, 1e-4)
        A[count, (2*k, 2*k+1)] = np.array([1,0])
        b[count] = 0.4*barrier_gain*h**3
        count += 1

        #Neg X
        h = -boundary_points[0] - safety_radius/2 + X[0,k]
        h = max(h, 1e-4)
        A[count, (2*k, 2*k+1)] = -np.array([1,0])
        b[count] = 0.4*barrier_gain*h**3
        count += 1

    #NETWORK BARRIERS
    radii = utils.communication_radius_list()
    GAMMA_network = 10.0

    returning_agents_indices = []
    for agent in env.agents:
        if agent.state == "Returning":
            returning_agents_indices.append(agent.id)

    A_network = np.zeros((N, 2*N))
    b_network = np.zeros(N)

    for i, row in enumerate(A_network):
        pos_i = X[:,i].reshape(2,1)
        diffs = X - pos_i
        dists = np.linalg.norm(diffs, axis=0)
        dists[i] = np.inf
        for idx in returning_agents_indices:
            dists[idx] = np.inf

        j = np.argmin(dists)

        ACTIVATION_THRESHOLD = radii[i] / 1.2

        if dists[j] <= radii[i] and i not in returning_agents_indices and dists[j] >= ACTIVATION_THRESHOLD:
            h = radii[i]**2 - dists[j]**2
            h = max(h, 1e-4)
            b_network[i] = -GAMMA_network*h**3

            A_network[i, i*2] = 2*(X[0,i]-X[0,j])
            A_network[i, i*2 + 1] = 2*(X[1,i]-X[1,j])
            A_network[i, j*2] = -2*(X[0,i]-X[0,j])
            A_network[i, j*2 + 1] = -2*(X[1,i]-X[1,j])

    #OBSTACLE BARRIERS
    num_obstacles = len(utils.OBSTACLE_LOCATIONS)
    num_obstacle_constraints = N*num_obstacles

    A_ob = np.zeros((num_obstacle_constraints, 2*N))
    b_ob = np.zeros((num_obstacle_constraints))

    ob_barrier_gain = barrier_gain

    count = 0
    for i in range(N):
        for j in range(num_obstacles):
            ob_x, ob_y = utils.OBSTACLE_LOCATIONS[j]["center"]
            sr = (safety_radius + utils.OBSTACLE_LOCATIONS[j]["radius"])/2

            h = (X[0,i] - ob_x)**2 + (X[1,i] - ob_y)**2 - sr**2
            h = max(h, 1e-4)
            b_ob[count] = ob_barrier_gain*h**3
            A_ob[count, 2*i]   = -2*(X[0,i] - ob_x)
            A_ob[count, 2*i+1] = -2*(X[1,i] - ob_y)
            count += 1

    A = matrix(np.vstack((A, A_network, A_ob)))
    b = matrix(np.hstack((b, b_network, b_ob)))

    if use_lambda:
        total_vars = 2 * N
        P = np.zeros((total_vars, total_vars))

        for i in range(N):
            lambda_i = lambda_values[i]
            u_ph_i = U[:, i].reshape(2, 1)

            P[2*i : 2*(i+1), 2*i : 2*(i+1)] += 2 * lambda_i * np.eye(2)
            Q_i = np.outer(u_ph_i, u_ph_i)
            P[2*i : 2*(i+1), 2*i : 2*(i+1)] += 2 * (1 - lambda_i) * Q_i

        P += 1e-6 * np.eye(total_vars)
    else:
        P = H

    norms = np.linalg.norm(U, 2, 0)
    idxs_to_normalize = (norms > magnitude_limit)
    U[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

    f = -2*np.reshape(U, (2*N,1), order='F')
    b = np.reshape(b, (len(b),1), order='F')

    solvers.options['abstol'] = 1e-10
    solvers.options['reltol'] = 1e-10
    solvers.options['feastol'] = 1e-10

    result = solvers.qp(matrix(P), matrix(f), matrix(A), matrix(b))

    # Check max constraint residual to fallback if needed
    residual = np.max(np.array(A @ np.array(result['x']).flatten() - b))
    if residual > 1e-5:
        print(f"[QP fallback] Constraint violation detected: {residual:.2e}, using fallback")
        f_fallback = -2 * np.reshape(U, (2*N,), order='F')
        result = solvers.qp(matrix(H), matrix(f_fallback), matrix(A), matrix(b))

    return np.reshape(result['x'], (2, N), order='F')
