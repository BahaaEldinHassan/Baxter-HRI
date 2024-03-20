import os
import numpy as np

data = []

## Loads the desired file & stores the file as a list ##
for line in list(np.loadtxt("trajectory_6.dat", delimiter=" ")):
    line = list(line)
    line[1] = -line[1]
    data.append(line)

print(np.array(data))

## Get the file path of this file ##
filenames = []
files = os.listdir(".")
## Sort the file names in alphabetical order ##
files.sort()
## For each file that is found ##
for filename in files:
    ## Select only the files begining with trajectory_ & have a file extension of .dat ##
    if filename.startswith("trajectory_") and filename.endswith(".dat"):
        ## Store those file names ##
        filenames.append(filename)

## Get the last found file of the trajectory files & increment the number by 1 ##
trajectoryFilename = str(int(str(filenames[-1]).split(".")[0].split("_")[1])+1)
## Print what the file name will be for the new trajectory to be stored in ##
print("\nTrajectory was saved to the file trajectory_"+trajectory+'.dat')
## Save the recored points of the trajectory in a text file with the extension .dat ##
np.savetxt("trajectory_"+trajectoryFilename+'.dat', Pos)