# Import packages and functions needed
import sys
import time as tm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import optimize, linalg
from netCDF4 import Dataset
# from mpl_toolkits.basemap import Basemap

from func_data import filter_month, yearbreak
from func_tests import test_kmeans, test_kmeans_PCA, test_kmeans_lowpass, \
                        test_kmeans_pers


###### IMPORT DATA ######
"The dataset consist of 39 years of daily data of geopotential height for \
December till March (around 4800 timesteps) on a grid of the Euro-Atlantic \
section (20-80N, 90W-30E or 30-87.5N, 80W-40E, 2.5 degrees), so 25/24 x 49 \
gridpoints."
data = Dataset('geo500_winter_all.nc', mode = 'r')          #20-80N, 90W-30E

# See what the keys are to get the variables
print( data.variables.keys() )

# Extract the data needed
lons = data.variables['longitude'][:]
lats = data.variables['latitude'][:]
time = data.variables['time'][:]
geo_all = data.variables['z'][:]

# Close the dataset
data.close()


###### DATA WEIGHTING ######
"On a longitude-latitude grid there are more data points at higher latitudes, \
this needs to be compensated by putting more weight on lower latitudes. The \
standard way to do this is by the square-root of the cosine of latitude."

# Create the weights for latitude and its sum
weights = np.sqrt( np.cos(lats * np.pi/180) )
weights_total = np.sum(weights) * len(lons)

# Create weighted data
geo_sc = geo_all * weights[:, np.newaxis]


###### DATA SELECTION ######
"Select only the months of data of interest (mainly DJFM) and compute the \
average."

# Select the months to be considered (0=NDJFM, 1=DJFM, 2=DJF)
months = 1

# Select the data for those months
geo = filter_month( geo_sc, months )
time_nr, lat_nr, lon_nr = geo.shape
print( geo.shape )

# Get the average background state for the full dataset and the deviations
geo_av = np.average( geo, axis=0 )
dev = geo - geo_av


###### DATA PARTITION ######
"Create partitions of the data to verify the clusters found and to study \
non-stationarity in the dataset."

# Create an array for the year breaks
winter_diff = yearbreak( time_nr, months )
wdl = len(winter_diff)

# Create a partition in odd and even years
geo_odd = geo[winter_diff[0]:winter_diff[1]]
geo_even = geo[winter_diff[1]:winter_diff[2]]
for i in range(2,wdl-1):
    if (i % 2) == 0:
        geo_odd = np.append( geo_odd, geo[winter_diff[i]:winter_diff[i+1]], axis=0 )
    else:
        geo_even = np.append( geo_even, geo[winter_diff[i]:winter_diff[i+1]], axis=0 )

# and compute deviations from their average
dev_odd = geo_odd - np.average( geo_odd, axis=0 )
dev_even = geo_even - np.average( geo_even, axis=0 )
dev_oe = [dev_odd, dev_even]


###### CLUSTER SETTINGS ######
"Set the parameters to be used for the clustering. The number of clusters and \
persistent constraint need to be given when running the script."

# Set the number of PCs and time-filter values
PCA_nr = [5,10,15,20]
lowpass_nr = [5, 10]

# Get the number of clusters and persistence constraint from the input
k_nr = int( sys.argv[1] )
c_nr = int( sys.argv[2] )

# Set the cluster tolerance and number of tests
tol = 0.00001 /(k_nr**2)
tol_pca = 0.0001 /(k_nr**2)
test_nr = 500


###### RUN THE TESTS ######
"Run the tests with given tolerance and input k with and without persistence \
(C). This can be done for both the full dataset, as well as the partioned. "

# The tests for the full data set
start_loop = tm.time()
output0 = test_kmeans( dev, k_nr, tol, test_nr, "kmeans_r1")
for pca in PCA_nr:
    tol_pca0 = tol_pca * pca/20
    output1 = test_kmeans_PCA( dev, k_nr, tol_pca0, pca, test_nr, \
                "kmeans_PCA"+repr(pca) )
for lp in lowpass_nr:
    output2 = test_kmeans_lowpass( dev, k_nr, tol, lp, test_nr, \
                "kmeans_lowpass"+repr(lp) )
print( "The time taken to do all tests is " + repr( tm.time() - start_loop) +" sec." )

# The tests with persistence
start_loop = tm.time()
output0 = test_kmeans_pers( dev, k_nr, c_nr, tol, test_nr, "pers")
print( "The time taken to do all tests is " + repr( tm.time() - start_loop) +" sec.")

# The tests for the partioned data
data_part = dev_oe
start_loop = tm.time()
for i in range(len(data_part)):
    output0 = test_kmeans( data_part[i], k_nr, tol, test_nr, "kmeans_oe"+repr(i+1))
    for pca in PCA_nr:
        output1 = test_kmeans_PCA( data_part[i], k_nr, tol_pca, pca, test_nr, \
                    "kmeans_PCA"+repr(pca)+"_oe"+repr(i+1) )
    for lp in lowpass_nr:
        output2 = test_kmeans_lowpass( data_part[i], k_nr, tol, lp, test_nr, \
                    "kmeans_lowpass"+repr(lp)+"_oe"+repr(i+1) )
print( "The time taken to do all tests is " + repr( tm.time() - start_loop) +" sec.")
