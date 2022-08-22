#!/usr/bin/python3

"""
    Author: Amanda Camacho
    
    Executable to rum SET-RBM training with arbitrarily given training parameters
    Based on proof of concept implementation of Sparse Evolutionary Training (SET) by Decebal Constantin Mocanu et al.
"""



import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import dok_matrix
#the "sparseoperations" Cython library was tested in Ubuntu 16.04. Please note that you may encounter some "solvable" issues if you compile it in Windows.
import sparseoperations
import datetime
import scipy.io as sio
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()

dtset = parser.add_argument_group(title='Training datasets', 
                description="Give dataset paths and other relevant information")
dtset.add_argument("-d", "--data", type=str, nargs="+", required=True,
                   help="List path of the datasets used. If you give 2 arguments, it "
                        "will be assumed that the first is the train set and the "
                        "second the test set.")
dtset.add_argument("-n", "--dataset-name", type=str, required=True, 
                   help="Give dataset name (to be used in the output file names)\n"
                        "For example: COIL20, MNIST, etc.")

trnparams = parser.add_argument_group(title="Training parameters", 
                description="Give parameters for the RBM training method")
trnparams.add_argument("-s", "--seed", type=int, default=-1,
                       help="Set random seed value. If the argument is unused, no seed "
                            "will be set.")
trnparams.add_argument("-H", "--hidden-neurons", type=int, required=True,
                       help="Set the number of hidden neurons in the RBM")
trnparams.add_argument("-E", "--sparsity-level", "--epsilon", type=int, default=10, 
                       help="SET sparsity level") 
trnparams.add_argument("-b", "--batch-size", default=10, type=int, 
                       help="Training batch size")
trnparams.add_argument("-e", "--epochs", default=100, type=int, 
                       help="Number of training epochs")
trnparams.add_argument("-k", "--length-markov-chain", default=10, type=int, 
                       help="Length of Markov chain for Contrastive Divergence") 
trnparams.add_argument("-r", "--learning-rate", "--alpha", type=float,
                       help="Training learning rate")

outparams = parser.add_argument_group(title="Output arguments", 
                description="DEtermine what kinf of output will be given and its characteristics")
outparams.add_argument("-f", "--f-nll", "--frequency-performance-metrics", type=int, default=1,
                       help="Frequency of calculation of performance estimation methods "
                            "(f.e.: calculate NLL at every 10 epochs)")
# TODO: type of output file?




def contrastive_divergence_updates_Numpy(wDecay, lr, DV, DH, MV, MH, rows, cols, out):
    for i in range (out.shape[0]):
        s1=0
        s2=0
        for j in range(DV.shape[0]):
            s1+=DV[j,rows[i]]*DH[j, cols[i]]
            s2+=MV[j,rows[i]]*MH[j, cols[i]]
        out[i]+=lr*(s1/DV.shape[0]-s2/DV.shape[0])-wDecay*out[i]
    #return out

def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx

def createSparseWeights(epsilon,noRows,noCols):
    # generate an Erdos Renyi sparse weights mask
    weights=lil_matrix((noRows, noCols))
    for i in range(epsilon * (noRows + noCols)):
        weights[np.random.randint(0,noRows),np.random.randint(0,noCols)]=np.float64(np.random.randn()/20)
    print ("Create sparse matrix with ",weights.getnnz()," connections and ",(weights.getnnz()/(noRows * noCols))*100,"% density level")
    weights=weights.tocsr()
    return weights

class Sigmoid:
    @staticmethod
    def activation(z):

        return 1 / (1 + np.exp(-z))

    def activationStochastic(z):
        z=Sigmoid.activation(z)
        za=z.copy()
        prob=np.random.uniform(0,1,(z.shape[0],z.shape[1]))
        za[za>prob]=1
        za[za<=prob]=0
        return za


class SET_RBM:
    def __init__(self, noVisible, noHiddens,epsilon=10):
        self.noVisible = noVisible #number of visible neurons
        self.noHiddens=noHiddens # number of hidden neurons
        self.epsilon = epsilon  # control the sparsity level as discussed in the paper

        self.learning_rate = None #learning rate
        self.weight_decay = None #weight decay
        self.zeta = None  # the fraction of the weights removed

        self.W=createSparseWeights(self.epsilon,self.noVisible,self.noHiddens) # create weights sparse matrix
        self.bV=np.zeros(self.noVisible) #biases of the visible neurons
        self.bH = np.zeros(self.noHiddens) #biases of the hidden neurons

    def fit(self, X_train, X_test, batch_size,epochs,lengthMarkovChain=2,weight_decay=0.0000002,learning_rate=0.1,zeta=0.3, testing=True, 
            outputType="", f_nll=1, save_filename="test.txt"):

        # set learning parameters
        self.lengthMarkovChain=lengthMarkovChain #length of Markov chain for Contrastive Divergence
        self.weight_decay=weight_decay #weight decay
        self.learning_rate=learning_rate #learning rate
        self.zeta=zeta #control the fraction of weights removed
        
        # Determining what outputs to save
        save_fileRecon = save_filename
        outType = outType.lower()
        calcNLL = False
        calcRecErr = False
        # if outType == "all":
        #     calcNLL = True
        #     calcRecErr = True
        # TODO: Add 'all' option. Will need to rethink how to give the output filename
        elif outType == "nll":
            calcNLL = True
        elif outType == "reconerror":
            calcRecErr = True
        else:
            if outType != "":
                print(f"WARNING: Cannot save output of type '{outType}'. Will not save any output!")
            
        
        outF = None
        pid = os.getpid()
        if calcNLL:
            # Open NLL output file 
            outF = open(save_filename, "w")
            outF.write(f"# NLL through RBM training for MNIST data set with connectivity pattern of type set (parameter {self.eps})\n")
            outF.write(f"# CD-{self.lengthMarkovChain}. Seed {self.seed}, Batch size = {self.batch_size} and learning rate of {self.learnin_rate}\n")
            outF.write(",NLL")
            # TODO: Check if there is a self.seed
            
            # First data point: NLL before training
            self.save(f"tmp_{pid}.rbm")
            os.system(f"./ais_estimator.sh {pid}")
            
            with open(f"lnZ_{pid}.txt", 'r') as lnF:
                    nll = float(lnF.readline())
                    outF.write(f"0,{nll}\n")
            

        minimum_reconstructin_error=100000
        metrics=np.zeros((epochs,2))
        reconstruction_error_train=0

        for i in range (epochs):
            # Shuffle the data
            seed = np.arange(X_train.shape[0])
            np.random.shuffle(seed)
            x_ = X_train[seed]

            # training
            t1 = datetime.datetime.now()
            for j in range(x_.shape[0] // batch_size):
                k = j * batch_size
                l = (j + 1) * batch_size
                reconstruction_error_train+=self.learn(x_[k:l])
            t2 = datetime.datetime.now()

            reconstruction_error_train=reconstruction_error_train/(x_.shape[0] // batch_size)
            metrics[i, 0] = reconstruction_error_train
            print ("\nSET-RBM Epoch ",i)
            print ("Training time: ",t2-t1,"; Reconstruction error train: ",reconstruction_error_train)
            
            # Calculate and save NLL at current epoch
            if calcNLL:
                if (i+1)%f_nll == 0:
                    self.save(f"tmp_{pid}.rbm")
                    os.system(f"./ais_estimator.sh {pid}")

                    with open(f"lnZ_{pid}.txt", 'r') as lnF:
                        nll = float(lnF.readline())
                        outF.write(f"{i+1},{nll}\n")
            
            # test model performance on the test data at each epoch
            # this part is useful to understand model performance and can be commented for production settings
            if (testing):
                t3 = datetime.datetime.now()
                reconstruction_error_test=self.reconstruct(X_test)
                t4 = datetime.datetime.now()
                metrics[i, 1] = reconstruction_error_test
                minimum_reconstructin_error = min(minimum_reconstructin_error, reconstruction_error_test)
                print("Testing time: ", t4 - t3, "; Reconstruction error test: ", reconstruction_error_test,"; Minimum reconstruction error: ", reconstruction_error_test)

            # change connectivity pattern
            t5 = datetime.datetime.now()
            if (i < epochs - 1):
                self.weightsEvolution(addition=True)
            else:
                if (i == epochs - 1): #during the last epoch just connections removal is performed. We did not add new random weights to favour statistics on the connections
                    self.weightsEvolution(addition=False)
            t6 = datetime.datetime.now()
            print("Weights evolution time ", t6 - t5)

            #save performance metrics values in a file
            if calcRecErr:
                np.savetxt(save_filename,metrics)
                
        if calcNLL:
            outF.close()

    def runMarkovChain(self,x):
        self.DV=x
        self.DH=self.DV@self.W  + self.bH
        self.DH=Sigmoid.activationStochastic(self.DH)

        for i in range(1,self.lengthMarkovChain):
            if (i==1):
                self.MV = self.DH @ self.W.transpose() + self.bV
            else:
                self.MV = self.MH @ self.W.transpose() + self.bV
            self.MV = Sigmoid.activation(self.MV)
            self.MH=self.MV@self.W  + self.bH
            self.MH = Sigmoid.activationStochastic(self.MH)

    def reconstruct(self,x):
        self.runMarkovChain(x)
        return (np.mean((self.DV-self.MV)*(self.DV-self.MV)))

    def learn(self,x):
        self.runMarkovChain(x)
        self.update()
        return (np.mean((self.DV - self.MV) * (self.DV - self.MV)))

    def getRecontructedVisibleNeurons(self,x):
        #return recontructions of the visible neurons
        self.reconstruct(x)
        return self.MV

    def getHiddenNeurons(self,x):
        # return hidden neuron values
        self.reconstruct(x)
        return self.MH


    def weightsEvolution(self,addition):
        # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
        # TODO: this method could be seriously improved in terms of running time using Cython
        values=np.sort(self.W.data)
        firstZeroPos = find_first_pos(values, 0)
        lastZeroPos = find_last_pos(values, 0)

        largestNegative = values[int((1-self.zeta) * firstZeroPos)]
        smallestPositive = values[int(min(values.shape[0] - 1, lastZeroPos + self.zeta * (values.shape[0] - lastZeroPos)))]

        wlil = self.W.tolil()
        wdok = dok_matrix((self.noVisible,self.noHiddens),dtype="float64")

        # remove the weights closest to zero
        keepConnections=0
        for ik, (row, data) in enumerate(zip(wlil.rows, wlil.data)):
            for jk, val in zip(row, data):
                if (((val < largestNegative) or (val > smallestPositive))):
                    wdok[ik,jk]=val
                    keepConnections+=1

        # add new random connections
        if (addition):
            for kk in range(self.W.data.shape[0]-keepConnections):
                ik = np.random.randint(0, self.noVisible)
                jk = np.random.randint(0, self.noHiddens)
                while ((wdok[ik,jk]!=0)):
                    ik = np.random.randint(0, self.noVisible)
                    jk = np.random.randint(0, self.noHiddens)
                wdok[ik, jk]=np.random.randn() / 20

        self.W=wdok.tocsr()

    def update(self):
        #compute Contrastive Divergence updates
        self.W=self.W.tocoo()
        sparseoperations.contrastive_divergence_updates_Cython(self.weight_decay, self.learning_rate, self.DV, self.DH, self.MV, self.MH, self.W.row, self.W.col, self.W.data)
        # If you have problems with Cython please use the contrastive_divergence_updates_Numpy method by uncommenting the line below and commenting the one above. Please note that the running time will be much higher
        #contrastive_divergence_updates_Numpy(self.weight_decay, self.learning_rate, self.DV, self.DH, self.MV, self.MH, self.W.row, self.W.col, self.W.data)

        # perform the weights update
        # TODO: adding momentum would make learning faster
        self.W=self.W.tocsr()
        self.bV=self.bV+self.learning_rate*(np.mean(self.DV,axis=0)-np.mean(self.MV,axis=0))-self.weight_decay*self.bV
        self.bH = self.bH + self.learning_rate * (np.mean(self.DH, axis=0) - np.mean(self.MH, axis=0)) - self.weight_decay * self.bH

if __name__ == "__main__":
    # Organizing rexeived arguments
    args = parser.parse_args()
    
    print(args)

    trainData = args.data[0]
    hasTestData = len(args.data) > 1
    if hasTestData:
        testData = args.data[1]
    else:
        testData = None
    dataset = args.dataset_name

    seed = args.seed
    H = args.hidden_neurons
    eps = args.sparsity_level
    bsize = args.batch_size
    epochs = args.epochs
    k = args.length_markov_chain
    alpha = args.learning_rate
    f_nll = args.frequency_performance_metrics
    
    filebase = f"{dataset}_set-{eps}_H{H}_CD-{k}_lr{alpha}_mBatch{bsize}_iter{epochs}"
    

    # Setting random seed
    if seed != -1:
        filebase += f"_run{seed}"
        np.random.seed(seed)


    # Loading data      (NOTE: I assume the data is binary!)
    mat = sio.loadmat(trainData) # Example: data/COIL20.mat
    X_train = mat['X']
    Y_train = mat['Y']  # NOTE: The labels are not used yet
    
    X_test, Y_test = None, None
    if hasTestData:
        mat = sio.loadmat(testData)
        X_test = mat['X']
        Y_test = mat['Y']  


    # Creating SET-RBM
    setrbm=SET_RBM(X_train.shape[1],noHiddens=H,epsilon=eps)

    # train SET-RBM
    setrbm.fit(X_train, X_test, 
               batch_size=bsize, epochs=epochs, 
               lengthMarkovChain=k, weight_decay=0, learning_rate=alpha, 
               zeta=0.3, testing=hasTestData, 
               "nll", save_filename=f"Results/nll_{filebase}.csv", f_nll)   # f"Results/reconErr_{filebase}.txt"

    # get reconstructed data
    # please note the very very small difference in error between this one and the one computing during training. This is the (insignificant) effect of the removed weights which are closest to zero
    if hasTestData:
        reconstructions=setrbm.getRecontructedVisibleNeurons(X_test)
        print ("\nReconstruction error of the last epoch on the testing data: ",np.mean((reconstructions-X_test)*(reconstructions-X_test)))

    # # get hidden neurons values to be used, for instance, with a classifier
    # hiddens=setrbm.getHiddenNeurons(X_test)

