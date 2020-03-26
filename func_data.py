# Import packages needed
import numpy as np

### Filter November out of the data ###
"The data (nov-mar) is selected based on the period that ones wants to consider. \
Setting the filter_index to 0 gives all data, 1 gets rid of November and 2 of \
November and March."

def filter_month( geo, filter_index ):

    # Get dimensions from geo
    time_nr = geo.shape[0]
    lat_nr = geo.shape[1]
    lon_nr = geo.shape[2]

    # Create an array with equidistant points for time and the year breaks
    time_r = np.arange(0, time_nr)
    nov = 30;   dec = 31;   jan = 31;   feb = 28;   mar = 31;
    feb_leap = feb+1
    days = nov+dec+jan+feb+mar
    leap = days+1
    winter_diff = [0,leap]
    for y in range(1981,2019):
        if int(y/4) == y/4 :
            winter_diff = np.append( winter_diff, winter_diff[-1] + leap )
        else:
            winter_diff = np.append( winter_diff, winter_diff[-1] + days )

    if filter_index == 0:       # All months
        geo_winter = geo
    elif filter_index == 1:     # No November
        winter_ind = []
        for i in range(0, len(winter_diff)-1):
            if winter_diff[i+1] - winter_diff[i] == 152:    # Leap years
                winter_ind = np.append( winter_ind, range(winter_diff[i]+nov, \
                                winter_diff[i]+nov+dec+jan+feb_leap+mar) )
            else:
                winter_ind = np.append( winter_ind, range(winter_diff[i]+nov, \
                                winter_diff[i]+nov+dec+jan+feb+mar) )
    elif filter_index == 2:     # No November and March
        winter_ind = []
        for i in range(0, len(winter_diff)-1):
            if winter_diff[i+1] - winter_diff[i] == 152:    # Leap years
                winter_ind = np.append( winter_ind, range(winter_diff[i]+nov, \
                                winter_diff[i]+nov+dec+jan+feb_leap) )
            else:
                winter_ind = np.append( winter_ind, range(winter_diff[i]+nov, \
                                winter_diff[i]+nov+dec+jan+feb) )
    else:
        print("Give a correct filter entry.")

    # If filtering, select geopotential data
    if filter_index != 0:
        geo_winter = np.zeros(( len(winter_ind), lat_nr, lon_nr ))
        for i in range(0, len(winter_ind) ):
            geo_winter[i] = geo[int(winter_ind[i])]

    return geo_winter


### Create a list for the begin and end of each winter ###
"A list which contains the numbers of the datapoints where each winter ends is \
created from the total length of the dataset (time_nr) and the number of months \
in each winter, given by the filter_index."

def yearbreak( time_nr, filter_index ):

    # Set the time length and winter length
    if filter_index == 0:
        days = 30 + 31 + 31 + 28 + 31
    elif filter_index == 1:
        days = 31 + 31 + 28 + 31
    elif filter_index == 2:
        days = 31 + 31 + 28
    else:
        print("Give a correct filter entry.")
    leap = days+1

    # Create an array with equidistant points for time and the year breaks
    winter_diff = [0,leap]
    for y in range(1981,2019):
        if int(y/4) == y/4 :
            winter_diff = np.append( winter_diff, winter_diff[-1] + leap )
        else:
            winter_diff = np.append( winter_diff, winter_diff[-1] + days )

    return winter_diff


### A low-pass filter to get rid of high-frequency oscillations ###
"Apply a low-pass filter to the data of geopotential height fields. Given the \
data and cut-off frequency (given by days), the output is a filtered dataset. \
Be aware that boundary effects occur."

def lowpass( data, cut_day ):

    # Get the cut-off frequency
    cut = 1/cut_day

    # Set the band-width filter according to the length of the dataset
    N = data.shape[0]
    if not N % 2: N += 1  # Make sure that N is odd.
    n = np.arange(N)

    # Compute sinc filter.
    filter = np.sinc(2 * cut * (n - (N - 1) / 2))

    # Compute Blackman window.
    window = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + \
                0.08 * np.cos(4 * np.pi * n / (N - 1))

    # Multiply sinc filter with window.
    filter = filter * window

    # Normalize to get unity gain.
    filter = filter / np.sum(filter)

    lat_nr = data.shape[1];     lon_nr = data.shape[2]
    data_filtered = np.zeros((data.shape[0], lat_nr, lon_nr))
    for i in range( lat_nr ):
        for j in range( lon_nr ):
            data_filtered[:,i,j] = np.convolve(data[:,i,j], filter, 'same')[0:data.shape[0]]

    return data_filtered


### A function to transform the field to its principal components ###
"A function to transform the geopotential height field (data) to its PCA_nr \
principal components. The output are the principal components, as well as the \
matrix for tranforming back to the full field and the explained variance for \
each PCA."

def PCA_analysis( data, PCA_nr ):

    # Get dimenions from data
    ( time_nr, lat_nr, lon_nr ) = data.shape

    # Get covariance matrix and its eigenvalues and vectors
    dev_list = np.reshape( data, (time_nr, lat_nr*lon_nr,1) )[:,:,0]
    cor_mat = np.corrcoef( dev_list.T )
    eig_vals, eig_vecs = np.linalg.eig(cor_mat)

    # Make a list of (eigenvalue, eigenvector) tuples (sorted)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    def sort0(eig):
        return eig[0]
    eig_pairs = sorted( eig_pairs, key=sort0, reverse=True)

    # Get the explained variance and plot
    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]

    # Get the PCA evolution in time
    matrix = np.hstack([(eig_pairs[i][1].reshape(lat_nr*lon_nr,1)) for i in range(PCA_nr)] )
    PCA = dev_list.dot( matrix )

    return PCA, matrix, var_exp
