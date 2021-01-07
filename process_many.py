import numpy as np
import sklearn
from sklearn import preprocessing

import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u

import lightkurve as lk

import json

gBINS = 1000
lBINS = 100

labels_file = "C:/Users/wnbau/Documents/Programming Projects/NeuralNetworks/Astronomy/files/all_labels.json"
labels_dict = {}

with open(labels_file) as file: 
    labels_dict = json.load(file) 

# get ids
file_paths = np.load("C:/Users/wnbau/Documents/Programming Projects/NeuralNetworks/Astronomy/unprocessed/ids.npy")
#file_paths = ["5733076", "72034540"]

file_length = len(file_paths)
count = 0
for file in file_paths:
    lc_file = "C:/Users/wnbau/Documents/Programming Projects/NeuralNetworks/Astronomy/unprocessed/" + file + "_lc.fits"
    dvt_file = "C:/Users/wnbau/Documents/Programming Projects/NeuralNetworks/Astronomy/unprocessed/" + file + "_dvt.fits"

    # give fits info from file
    #fits.info(lc_file)

    # open fits file with astropy
    data = fits.open(lc_file)

    # get label from TIC ID
    sim_data = labels_dict[data[0].header['OBJECT'][4:]]
    #print(sim_data)

    # get fold periods from dvt file
    #fits.info(dvt_file)
    dvt_data = fits.open(dvt_file)

    # get time datapoints from fits file
    TIME = np.array(data[1].data['TIME'])
    TIMECORR = np.array(data[1].data['TIMECORR'])

    # adjust time with time correction
    TIME = np.subtract(TIME, TIMECORR)
    TIME = np.subtract(TIME, dvt_data[1].header['TEPOCH'])

    # get flux datapoints from fits file (PDC corrected)
    PDCSAP_FLUX = np.array(data[1].data['PDCSAP_FLUX'])

    # remove NaNs from TIME
    NAN_ARRAY = np.logical_or(np.isinf(TIME), np.isinf(PDCSAP_FLUX))
    NAN_ARRAY = np.logical_or(NAN_ARRAY, np.isnan(PDCSAP_FLUX))
    NAN_ARRAY = np.logical_or(NAN_ARRAY, np.isnan(TIME))

    TIME = TIME[~NAN_ARRAY]
    PDCSAP_FLUX = PDCSAP_FLUX[~NAN_ARRAY]

    # initialize light curve objects
    global_lc = lk.LightCurve(time=TIME, flux=PDCSAP_FLUX)

    # get number of elements
    global_LENGTH = len(TIME)

    # preprocess
    global_lc = global_lc.flatten(global_LENGTH + 1)
    global_lc = global_lc.fold(dvt_data[1].header['TPERIOD'])

    local_lc = global_lc[(global_lc.phase > -2.5 * (dvt_data[1].header['TDUR']/24.0/dvt_data[1].header['TPERIOD'])) & (global_lc.phase < 2.5 * (dvt_data[1].header['TDUR']/24.0/dvt_data[1].header['TPERIOD']))]
    local_LENGTH = len(local_lc)
    local_lc = local_lc.fold(dvt_data[1].header['TPERIOD'])

    # bin
    global_lc = global_lc.bin(bins=gBINS)
    local_lc = local_lc.bin(bins=lBINS)

    np.save("./processed/" + data[0].header['OBJECT'][4:] + "_g", np.array(global_lc.flux))
    np.save("./processed/" + data[0].header['OBJECT'][4:] + "_l", np.array(local_lc.flux))

    if (sim_data["type"] == "variation"):
        np.save("./processed/" + data[0].header['OBJECT'][4:] + "_y", np.array([0.0,0.0,0.0,1.0]))
        np.save("./processed/" + data[0].header['OBJECT'][4:] + "_s", np.array([0.0,1.0]))
    elif (sim_data["type"] == "planet"):
        np.save("./processed/" + data[0].header['OBJECT'][4:] + "_y", np.array([0.0,0.0,1.0,0.0]))
        np.save("./processed/" + data[0].header['OBJECT'][4:] + "_s", np.array([1.0,0.0]))
    elif (sim_data["type"] == "backeb"):
        np.save("./processed/" + data[0].header['OBJECT'][4:] + "_y", np.array([1.0,0.0,0.0,0.0]))
        np.save("./processed/" + data[0].header['OBJECT'][4:] + "_s", np.array([0.0,1.0]))
    else:
        np.save("./processed/" + data[0].header['OBJECT'][4:] + "_y", np.array([0.0,1.0,0.0,0.0]))
        np.save("./processed/" + data[0].header['OBJECT'][4:] + "_s", np.array([0.0,1.0]))
    
    count += 1
    if count % 50 == 0:
        print(str(count) + "/" + str(file_length))