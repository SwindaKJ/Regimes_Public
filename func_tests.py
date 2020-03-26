# Import packages and functions needed
import time as tm
import numpy as np
from scipy import linalg

# Import functions
from func_data import filter_month, yearbreak, lowpass, PCA_analysis
from func_clustering import distance, kmeans, min_theta, min_gamma, kmeans_pers
from func_PCA_clustering import distance_PCA, kmeans_PCA, min_theta_PCA, \
                                    min_gamma_PCA, kmeans_pers_PCA


### A loop to do the k-means clustering a number of times ###
"A function to do the kmeans clustering test_nr times and compare the results. \
Input is the data, the number of clusters and tolerance, together with the name \
under which to save the result."

def test_kmeans( data, k_nr, tol, test_nr, name ):

    # Get dimensions
    (time_nr, lat_nr, lon_nr) = data.shape

    # Initilize the output arrays
    theta_test = np.zeros(( test_nr, k_nr, lat_nr, lon_nr ))
    seq_test = np.zeros(( test_nr, time_nr ))
    dist_test = np.zeros(( test_nr, k_nr, time_nr ))

    # Repeat the clustering test_nr times with different initial conditions
    start_loop = tm.time()
    for test in range(0, test_nr):
        theta_test[test], seq_test[test], dist_test[test] = kmeans( data, k_nr, tol)
    print( "The time taken to do "+repr(test_nr)+" loops is " + repr( tm.time() - start_loop) )

    np.savez( "tests/Test_k"+repr(k_nr)+"_"+name+"_"+repr(test_nr), \
                theta=theta_test, sequence=seq_test, distance=dist_test )

    return print( "The test is completed and saved as \"Test_k"+repr(k_nr)+"_"+name+"_"+repr(test_nr)+".npz\".")


### A loop to do the k-means clustering with persistence a number of times ###
"A function to do the kmeans clustering with a persistence constraint test_nr \
times and compare the results. Input is the data, the number of clusters and \
tolerance, together with the name under which to save the result."

def test_kmeans_pers( data, k_nr, c_nr, tol, test_nr, name ):

    # Get dimensions
    (time_nr, lat_nr, lon_nr) = data.shape

    # Initilize the output arrays
    theta_test = np.zeros(( test_nr, k_nr, lat_nr, lon_nr ))
    gam_test = np.zeros(( test_nr, k_nr, time_nr ))
    seq_test = np.zeros(( test_nr, time_nr ))
    dist_test = np.zeros(( test_nr, k_nr, time_nr ))

    # Repeat the clustering test_nr times with different initial conditions
    start_loop = tm.time()
    for test in range(0, test_nr):
        theta_test[test], gam_test[test], seq_test[test], dist_test[test] = kmeans_pers( data, k_nr, c_nr, tol)
    print( "The time taken to do "+repr(test_nr)+" loops is " + repr( tm.time() - start_loop) )

    np.savez( "tests/Test_k"+repr(k_nr)+"_C"+repr(c_nr)+"_"+name+"_"+repr(test_nr), \
                theta=theta_test, sequence=seq_test, distance=dist_test )

    return print( "The test is completed and saved as \"Test_k"+repr(k_nr)+"_C"+repr(c_nr)+"_"+name+"_"+repr(test_nr)+"\".npz .")


### A loop to do the k-means clustering a number of times for low pass filtered data ###
"A function to do the kmeans clustering for lowpass filtered data test_nr times \
and compare the results. Input is the data, the number of clusters, the \
tolerance, and the filter frequency in days, together with the name under which \
to save the result."

def test_kmeans_lowpass( data, k_nr, tol, bnd_day, test_nr, name ):

    # Get dimensions
    (time_nr, lat_nr, lon_nr) = data.shape

    # Apply a low-pass filter to the data
    geo_filtered = lowpass( data, bnd_day )

    # Get rid of boundary effects of the filter
    bnd_eff = 4*bnd_day
    data_filtered = np.copy(data)
    data_filtered[bnd_eff:-bnd_eff] = geo_filtered[bnd_eff:-bnd_eff]

    # Initilize the output arrays
    theta_test = np.zeros(( test_nr, k_nr, lat_nr, lon_nr ))
    seq_test = np.zeros(( test_nr, time_nr ))
    dist_test = np.zeros(( test_nr, k_nr, time_nr ))

    # Repeat the clustering test_nr times with different initial conditions
    start_loop = tm.time()
    for test in range(0, test_nr):
        theta_test[test], seq_test[test], dist_test[test] = kmeans( data_filtered, k_nr, tol)
    print( "The time taken to do "+repr(test_nr)+" loops is " + repr( tm.time() - start_loop) )

    np.savez( "tests/Test_k"+repr(k_nr)+"_"+name+"_"+repr(test_nr), \
                theta=theta_test, sequence=seq_test, distance=dist_test )

    return print( "The test is completed and saved as \"Test_k"+repr(k_nr)+"_"+name+"_"+repr(test_nr)+".npz\".")


### A loop to do the clustering for PCA a number of times ###
"A function to do the kmeans clustering for the PCAs of the data test_nr times \
and compare the results. Input is the data, the number of clusters, the \
tolerance, and the number of PCAs to consider, together with the name under \
which to save the result."

def test_kmeans_PCA( data, k_nr, tol, PCA_nr, test_nr, name ):

    # Get dimensions
    ( time_nr, lat_nr, lon_nr ) = np.shape(data)

    # Get the principal components
    PCA, PCA_mat, var_exp = PCA_analysis( data, PCA_nr )

    # Initilize the output arrays
    theta_test = np.zeros(( test_nr, k_nr, lat_nr, lon_nr ))
    theta_PCAtest = np.zeros(( test_nr, k_nr, PCA_nr ))
    seq_test = np.zeros(( test_nr, time_nr ))
    dist_test = np.zeros(( test_nr, k_nr, time_nr ))

    # Repeat the clustering test_nr times with different initial conditions
    start_loop = tm.time()
    for test in range(0, test_nr):
        theta_PCAtest[test], seq_test[test], dist_test[test] = kmeans_PCA( PCA, k_nr, tol)

        # Translate the PCAs back to fields
        theta_list = theta_PCAtest[test].dot(PCA_mat.T)
        theta_test[test] = np.reshape(theta_list, (k_nr, lat_nr, lon_nr))

    print( "The time taken to do "+repr(test_nr)+" loops is " + repr( tm.time() - start_loop) )

    np.savez( "tests/Test_k"+repr(k_nr)+"_"+name+"_"+repr(test_nr), \
                theta=theta_test, theta_PCA=theta_PCAtest, sequence=seq_test, \
                distance=dist_test )

    return print( "The test is completed and saved as \"Test_k"+repr(k_nr)+"_"+name+"_"+repr(test_nr)+".npz\" .")
