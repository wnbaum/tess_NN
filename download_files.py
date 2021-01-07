import wget
import json

import numpy as np

labels_file = "C:/Users/wnbau/Documents/Programming Projects/NeuralNetworks/Astronomy/files/all_labels.json"

labels_dict = {}

with open(labels_file) as file: 
    labels_dict = json.load(file) 

file_paths = list(labels_dict.keys())

#file_paths = file_paths[:5]

total = len(file_paths)
count = 1

valid_files = []

for file in file_paths:
    index = ((16-len(file))*'0') + file

    url = "https://archive.stsci.edu/missions/tess/ete-6/tid/" + index[0:2] + "/"+ index[2:5] + "/" + index[5:8] + "/" + index[8:11] + "/"

    print(str(count) + "/" + str(total) + " \n", end="")
    try:
        wget.download(url + "tess2019128220341-" + index + "-00011_dvt.fits", "C:/Users/wnbau/Documents/Programming Projects/NeuralNetworks/Astronomy/unprocessed/" + file + "_dvt.fits")
        wget.download(url + "tess2019128220341-" + index + "-0016-s_lc.fits", "C:/Users/wnbau/Documents/Programming Projects/NeuralNetworks/Astronomy/unprocessed/" + file + "_lc.fits")
        valid_files.append(file)
        print("\n")
    except:
        print("Likely missing .dvt\n")

    count += 1

np.save("C:/Users/wnbau/Documents/Programming Projects/NeuralNetworks/Astronomy/unprocessed/ids.npy", valid_files)
