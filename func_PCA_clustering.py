# Import packages needed
import numpy as np
import gurobipy as grb
import time as tm

### A function to compute the distance between two clusters ###
"For every timestep in the dataset the distance with the given cluster is \
computed following the elementwise L2 norm."

# Define the function
def distance_PCA( data, cluster ):

    diff = data - cluster                           # Compute the difference
    dist = np.linalg.norm( diff, axis=1 )           # Compute the norm of difference
    dist = dist /(data.shape[1])                    # Normalise

    return dist

### k-means clustering ###
"Implement the k-means clustering. Input is the full dataset, the desired number\
of clusters k_nr and the tolerance (tol). In every step the distance between the\
data and the different clusters is computed. Based on this distance each data-\
point is assigned to a cluster. The new clusters are computed as the average of \
all data belonging to that cluster. This iteration is continued until the result\
is not improving sufficiently anymore. The output are the different clusters, \
the switching sequence and the distance of the data to the clusters."

def kmeans_PCA( data, k_nr, tol ):

    # Get the data size
    ( time_nr, pca_nr ) = np.shape(data)

    # Set maximum number of steps
    step_max = 200

    # Create initial arrays
    dist = np.ones(( k_nr, time_nr ))           # The distance vector
    L_sum = sum( dist.flatten() ) /time_nr      # Set initial L
    hist_L = 0                                  # Initial history of L
    theta = np.random.randn( k_nr, pca_nr ) * np.amax( data )
        # The initial random clusters, scaled by the data

    # Set the counter
    j = 0

    # As long as the desired tolerance, or the maximum step number are not reached
    while( j < step_max and np.abs( L_sum - hist_L ) > tol ):

        # Set the maximum distance for comparison
        hist_L = L_sum

        # Get distance between the data and the clusters
        for k in range(0,k_nr):
            dist[k] = distance_PCA( data, theta[k] )

        # Create a sequence of which cluster is closest
        seq = np.argmin( dist, axis=0 )

        # Implement a back up if a cluster is not present
        for k in range(0, k_nr):
            if all( s!=k for s in seq):
                rand_int = np.random.randint(0,time_nr)
                seq[ rand_int ] = k
                print( 'Cluster ' + repr(k) + ' was not present in the data, \
                therefore a perturbation has been added at time ' \
                + repr(rand_int) )

        # Get the list of minimal distances to find L
        L_list = np.min( dist, axis=0 )
        L_sum = sum( L_list ) /time_nr      # Normalize by division by datasize

        # Update the clusters as the average of all data assigned to that cluster
        for k in range(0,k_nr):
            theta[k] = np.average( data[seq==k], axis=0 )

        # Update the counter
        j = j+1

    # Print information about the success of the optimalization
    if j == step_max:
        print( 'The maximum stepnumber (' + repr(step_max) + ') was reached.' )
    else:
        # Compute the final distances, sequence and L
        for k in range(0,k_nr):
            dist[k] = distance_PCA( data, theta[k] )
        seq = np.argmin( dist, axis=0 )
        L_list = np.min( dist, axis=0 )
        L_sum = sum( L_list ) /time_nr

        print( 'The result is not improving sufficiently anymore, after ' \
        + repr(j) + ' steps and the distance is ' + repr( L_sum ) )

    return theta, seq, dist


### A function to minimize the cluster-parameters given the weights ###
"Given the weights gamma, the number of clusters k, a tolerance, and initial \
clusters, this function finds the k clusters for which the distance with the \
data is minimal. It returns the optimal clusters as well as the corresponding \
switching sequence."

def min_theta_PCA( data, k_nr, tol, gam, theta_init ):

    # Get the time and spatial dimensions
    ( time_nr, pca_nr ) = np.shape(data)

    # Set initial arrays
    dist = np.ones(( k_nr, time_nr ))
    dist_wtd = dist
    theta = theta_init
    dist_max = 1

    # Set counter and maximum number of steps
    j = 0
    step_max = 100

    # As long as the difference with the next step is large
    while ( j < step_max and dist_max > tol ):

        # Set the maximum distance for comparison
        hist_max = sum( dist_wtd.flatten() )

        # Get distance between the data and the clusters
        for k in range(0,k_nr):                                 # Set the range
            dist[k] = distance_PCA( data, theta[k] )                # Compute norm

        # Multiply by the weights
        dist_wtd = np.multiply( gam, dist )

        # Compute the value of the minimization functional
        L = sum( dist_wtd.flatten() )
        # print( L )

        # Compute the weights for the averaging of the data
        dist_inv = np.zeros( dist_wtd.shape )
        dist_inv[ dist_wtd!=0 ] = 1/ dist_wtd[ dist_wtd!=0 ]
        # dist_inv[gam==0] = 0

        for k in range(0, k_nr):
            if all( d==0 for d in dist_inv[k]):
                rand_int = np.random.randint(0,time_nr)
                dist_inv[ k, rand_int ] = 1
                print( 'Cluster ' + repr(k) + ' was not present in the data, \
                therefore a perturbation has been added at time ' \
                + repr(rand_int) )
        weights_dist = dist_inv /sum( dist_inv )

        for k in range(0,k_nr):
            theta[k] = np.average( data, axis=0, weights=weights_dist[k] )

        # Compute difference with previous step
        dist_max = np.abs( L - hist_max )
        j = j+1

    # Print the counter and weigthed distance
    print( 'The counter for the theta minimization is ' + repr(j) + ' and the \
    maximum weighted distance is ' + repr(dist_max) )

    return theta, L, dist



### Minimize for the weights gamma given the cluster parameters ###
"Given the clusters, that is, the distance between each cluster and the data at \
every time step, we use gurobi to minimize for the weights gamma given a \
constraint on the persistence (c_nr)."

def min_gamma_PCA( dist, c_nr):

    # Get the number of timesteps and clusters from distance vector
    ( k_nr, time_nr ) = np.shape(data)

    # Set the model
    mod = grb.Model('MinGamma')

    # Create variables for the weights by their cluster and time
    weights = [ (k,t) for k in range(k_nr) for t in range(time_nr) ]
    gam = mod.addVars(weights, lb=0, ub=1, name = "gam")

    # Create additional variables needed for the absolute value constraints
    additional = [(k,t) for k in range(k_nr) for t in range(time_nr-1)]
    y1var = mod.addVars(additional, lb=-1, ub=1, name = "y1var" )
    y2var = mod.addVars(additional, lb=0, ub=1, name = "y2var" )

    # Set the objective function to minimize
    mod.setObjective( sum( dist[k,t]*gam[k,t] for k in range(k_nr) \
                        for t in range(time_nr) ), sense=grb.GRB.MINIMIZE )

    # Add the normalization and persistence constraints
    mod.addConstrs( (sum(gam[k,t] for k in range(k_nr)) == 1 for t in range(time_nr)), \
                    name = "normalization" )
    mod.addConstrs( ( y1var[k,t] == gam[k,t+1] - gam[k,t] \
                    for k in range(k_nr) for t in range(time_nr-1) ), \
                    name = "abs_y1" )
    mod.addConstrs( ( y2var[k,t] == grb.abs_(y1var[k,t]) \
                    for k in range(k_nr) for t in range(time_nr-1) ), \
                    name = "abs_y2" )
    mod.addConstr( (sum( y2var[k,t] for t in range(time_nr-1) \
                    for k in range(k_nr) ) <= c_nr ), name = "persistence")

    # Compute optimal solution
    mod.optimize()

    # Get the solution for gamma
    if mod.status == grb.GRB.Status.OPTIMAL:
        solution = mod.getAttr('x', gam)
        gam_new = np.zeros((k_nr, time_nr))
        for k in range(k_nr):
            for t in range(time_nr):
                gam_new[k,t] = solution[(k,t)]
    else:
        print( "Optimization failed" )

    return gam_new


### A function for the k-means clustering using the persistence constraint ###
"Given data, a number of clusters k_nr, a persistence constraint c_nr and a \
tolerance tol, clusters theta and corresponding weights gamma are computed \
using k-means clustering to optimize the clusters given the weights and linear \
programming to optimize the weights given the clusters. This is done \
iteratively until convergence. Next to the clusters theta and weights gamma, \
also the switching sequence and distance to the clusters for the data is \
included in the output."

def kmeans_pers_PCA( data, k_nr, c_nr, tol ):

    # Get time of dataset
    ( time_nr, pca_nr ) = np.shape(data)

    # Choose initial weights for the different clusters (in time)
    "The weights are in a k x time matrix, so for every timestep there is a certain\
    weight for each cluster, such that they sum to one at every time."
    gam_rand = np.random.rand( k_nr, time_nr )              # Create random matrix
    gam_sum = gam_rand.sum(axis=0)                          # Sum for each k
    gam = gam_rand / gam_sum[None,:]                        # Normalise to get weights

    # Do the minimalization for the first step with a random initial vector
    "The clusters are k list of PCAs."
    theta_init = np.random.rand( k_nr, pca_nr ) * np.amax(data)

    # Get distance between the data and the initial clusters
    "The distance between a cluster and a data point is a number, so the total \
    matrix is k x time."
    dist0 = np.ones(( k_nr, time_nr ))
    for k in range(0,k_nr):                                 # Set the range
        dist0[k] = distance( data, theta_init[k] )          # Computer norm

    # Compute the weighted functional
    L0 = sum( np.multiply( gam, dist0 ).flatten() )

    # Compute the first updated clusters
    theta, L, dist = min_theta_PCA( data, k_nr, tol, gam, theta_init )

    # Set the iteration counter
    j = 0
    step_max = 100

    # Time the loop to minimize L
    start_full = tm.time()
    while ( j < step_max and abs( L - L0 ) > tol ):

        # Set the updated reference value of the minimization functional
        L0 = L

        # Compute the next value for the weigths given the clusters
        gam = min_gamma_PCA( dist, c_nr )

        # Compute the next clusters for the new weights
        theta, L, dist = min_theta_PCA( data, k_nr, tol, gam, theta )
        print(L)

        # Update the counter
        j = j+1

    # Print the counter and weigthed distance
    print( 'The counter is ' + repr(j) + ' and the value of L is ' + repr(L) )

    # Get the end time
    end_full = tm.time()
    print( 'The time taken for the full loop is ' + repr( end_full - start_full ))

    # Get the sequence of the clusters
    seq = np.zeros( time_nr )
    for k in range(0,k_nr):
        seq[ gam[k]==1 ] = k

    return theta, gam, seq, dist
