"""
    Author: Amanda Camacho

    Script to convert a DATA file (used in the NCG project https://github.com/AmieOliveira/NCG.git) to MAT file
"""


import argparse
import numpy as np
import scipy.io as sio

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", required=True, help="Input (DATA) file path")
parser.add_argument("-o", "--output", help="Use if you want the name of the output file to be different than the input one")


if __name__ == "__main__":
    args = parser.parse_args()
    
    out = {}

    if args.input[-5:] != ".data":
        raise ValueError("File has unexpected extension. Should be a DATA file")

    with open(args.input, "r") as f:
        N = 0
        X = 0
        hasLab = False

        while True: # Read file characteristics
            line = f.readline()

            if not line:
                raise ValueError("File given is empty. Should be a DATA file")
            if line == "\n":
                break
            
            words = line.split(": ")
            
            if words[0] == "Number of examples":
                N = int(words[1])
            elif words[0] == "Example size":
                X = int(words[1])
            elif words[0] == "Has labels":
                if words[1] == "Yes\n":
                    hasLab = True

        out['X'] = np.ndarray((N, X))
        if hasLab:
            out['Y'] = np.ndarray((N, 1))
        

        for i in range(N):
            line = f.readline()
            if not line:
                print("WARNING: Less data points than expected according to the file description")
                break

            if hasLab:
                words = line.split(": ")
                if words[0] != "Label":
                    raise ValueError("File given does not have the correct format (no label in {i}th data point). Should be a DATA file")
                out['Y'][i] = int(words[1])
                
                line = f.readline()
                if not line:
                    raise ValueError("File given does not have the correct format (no data for {i}th data point). Should be a DATA file")

            words = line.split(" ")
            for j in range(X):
                out['X'][i,j] = int(words[j])


        outname = ""
        if args.output:
            outname = args.output
            if outname[-4:] != ".mat":
                outname += ".mat"
        else:
            parts = args.input.split("/")
            outname = parts[-1][:-4] + "mat"

        sio.savemat(outname, out)

