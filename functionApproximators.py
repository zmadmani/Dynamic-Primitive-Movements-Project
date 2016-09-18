import math
import numpy as np

#############################################################
# Function interpolation
#############################################################
def interpolate_function(S, parameters):
    f = parameters["f"]
    i = 0
    while(i < len(f) and f[i]["s"] > S):
        i = i + 1
    if(i == len(f)):
        return f[i-1]["value"]
    elif(i == 0):
        return f[i]["value"]
    else:
        return f[i-1]["value"] + (S - f[i-1]["s"])*(f[i]["value"]-f[i-1]["value"])/(f[i]["s"]-f[i-1]["s"])


##############################################################
# Implementing Gaussian Basis Functions
##############################################################
class GBF:
    def __init__(self, numFuncs, centers, widths):
        #Test if number of centers and widths match numFuncs
        if(len(centers) != numFuncs or len(widths) != numFuncs):
            raise Exception("Number of centers("+str(len(centers))+")/widths"+str(len(widths))+" provided doesn't match number of functions"+str(numFuncs)+".")
        self.w = np.empty([numFuncs])
        self.centers = centers
        self.widths = widths

    # Given funciton index i and s value returns f_i(s)
    def eval_function(self, i, s):
        return s*math.exp(-1*self.widths[i]*math.pow((s - self.centers[i]),2))

    # Arguments are 2 arrays of length M of the M input and output values
    def train(self, input, output):
        if(len(input) != len(output)):
            raise Exception("Number of inputs("+len(input)+") and outputs("+len(output)+") don't match")
        transformedInput = np.empty([len(input),len(self.w)])
        for i in range(len(input)):
            transformed = [self.eval_function(func,input[i]) for func in range(len(self.w))]
            transformedInput[i] = transformed
        self.w = np.linalg.lstsq(transformedInput, output)[0]

    def predict(self, input):
        transformedInput = np.asarray([self.eval_function(func,input) for func in range(len(self.w))])
        prediction = transformedInput.dot(self.w)
        return prediction
