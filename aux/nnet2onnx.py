import numpy as np 
import sys
import onnx
from onnx import helper, numpy_helper, TensorProto

def readNNet(nnetFile, withNorm=False):
    '''
    Read a .nnet file and return list of weight matrices and bias vectors
    
    Inputs:
        nnetFile: (string) .nnet file to read
        withNorm: (bool) If true, return normalization parameters
        
    Returns: 
        weights: List of weight matrices for fully connected network
        biases: List of bias vectors for fully connected network
    '''
    
    
        
    # Open NNet file
    f = open(nnetFile,'r')
    
    # Skip header lines
    line = f.readline()
    while line[:2]=="//":
        line = f.readline()
        
    # Extract information about network architecture
    record = line.split(',')
    numLayers   = int(record[0])
    inputSize   = int(record[1])

    line = f.readline()
    record = line.split(',')
    layerSizes = np.zeros(numLayers+1,'int')
    for i in range(numLayers+1):
        layerSizes[i]=int(record[i])

    # Skip extra obsolete parameter line
    f.readline()
    
    # Read the normalization information
    line = f.readline()
    inputMins = [float(x) for x in line.strip().split(",") if x]

    line = f.readline()
    inputMaxes = [float(x) for x in line.strip().split(",") if x]

    line = f.readline()
    means = [float(x) for x in line.strip().split(",") if x]

    line = f.readline()
    ranges = [float(x) for x in line.strip().split(",") if x]

    # Read weights and biases
    weights=[]
    biases = []
    for layernum in range(numLayers):

        previousLayerSize = layerSizes[layernum]
        currentLayerSize = layerSizes[layernum+1]
        weights.append([])
        biases.append([])
        weights[layernum] = np.zeros((currentLayerSize,previousLayerSize))
        for i in range(currentLayerSize):
            line=f.readline()
            aux = [float(x) for x in line.strip().split(",")[:-1]]
            for j in range(previousLayerSize):
                weights[layernum][i,j] = aux[j]
        #biases
        biases[layernum] = np.zeros(currentLayerSize)
        for i in range(currentLayerSize):
            line=f.readline()
            x = float(line.strip().split(",")[0])
            biases[layernum][i] = x

    f.close()
    
    if withNorm:
        return weights, biases, inputMins, inputMaxes, means, ranges
    return weights, biases

def writeNNet(weights,biases,inputMins,inputMaxes,means,ranges,fileName):
    '''
    Write network data to the .nnet file format
    Args:
        weights (list): Weight matrices in the network order 
        biases (list): Bias vectors in the network order
        inputMins (list): Minimum values for each input
        inputMaxes (list): Maximum values for each input
        means (list): Mean values for each input and a mean value for all outputs. Used to normalize inputs/outputs
        ranges (list): Range values for each input and a range value for all outputs. Used to normalize inputs/outputs
        fileName (str): File where the network will be written
    '''
    
    #Open the file we wish to write
    with open(fileName,'w') as f2:

        #####################
        # First, we write the header lines:
        # The first line written is just a line of text
        # The second line gives the four values:
        #     Number of fully connected layers in the network
        #     Number of inputs to the network
        #     Number of outputs from the network
        #     Maximum size of any hidden layer
        # The third line gives the sizes of each layer, including the input and output layers
        # The fourth line gives an outdated flag, so this can be ignored
        # The fifth line specifies the minimum values each input can take
        # The sixth line specifies the maximum values each input can take
        #     Inputs passed to the network are truncated to be between this range
        # The seventh line gives the mean value of each input and of all outputs
        # The eighth line gives the range of each input and of all outputs
        #     These two lines are used to map raw inputs to the 0 mean, unit range of the inputs and outputs
        #     used during training
        # The ninth line begins the network weights and biases
        ####################
        f2.write("// Neural Network File Format by Kyle Julian, Stanford 2016\n")

        #Extract the necessary information and write the header information
        numLayers = len(weights)
        inputSize = weights[0].shape[1]
        outputSize = len(biases[-1])
        maxLayerSize = inputSize
        
        # Find maximum size of any hidden layer
        for b in biases:
            if len(b)>maxLayerSize :
                maxLayerSize = len(b)

        # Write data to header 
        f2.write("%d,%d,%d,%d,\n" % (numLayers,inputSize,outputSize,maxLayerSize) )
        f2.write("%d," % inputSize )
        for b in biases:
            f2.write("%d," % len(b) )
        f2.write("\n")
        f2.write("0,\n") #Unused Flag

        # Write Min, Max, Mean, and Range of each of the inputs and outputs for normalization
        f2.write(','.join(str(inputMins[i])  for i in range(inputSize)) + ',\n') #Minimum Input Values
        f2.write(','.join(str(inputMaxes[i]) for i in range(inputSize)) + ',\n') #Maximum Input Values                
        f2.write(','.join(str(means[i])      for i in range(inputSize+1)) + ',\n') #Means for normalizations
        f2.write(','.join(str(ranges[i])     for i in range(inputSize+1)) + ',\n') #Ranges for noramlizations

        ##################
        # Write weights and biases of neural network
        # First, the weights from the input layer to the first hidden layer are written
        # Then, the biases of the first hidden layer are written
        # The pattern is repeated by next writing the weights from the first hidden layer to the second hidden layer,
        # followed by the biases of the second hidden layer.
        ##################
        for w,b in zip(weights,biases):
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    f2.write("%.5e," % w[i][j]) #Five digits written. More can be used, but that requires more more space.
                f2.write("\n")
                
            for i in range(len(b)):
                f2.write("%.5e,\n" % b[i]) #Five digits written. More can be used, but that requires more more space.


def normalizeNNet(readNNetFile, writeNNetFile=None):
    weights, biases, inputMins, inputMaxes, means, ranges = readNNet(readNNetFile,withNorm=True)
    
    numInputs = weights[0].shape[1]
    numOutputs = weights[-1].shape[0]
    
    # Adjust weights and biases of first layer
    for i in range(numInputs): weights[0][:,i]/=ranges[i]
    biases[0]-= np.matmul(weights[0],means[:-1])
    
    # Adjust weights and biases of last layer
    weights[-1]*=ranges[-1]
    biases[-1] *= ranges[-1]
    biases[-1] += means[-1]
    
    # Nominal mean and range vectors
    means = np.zeros(numInputs+1)
    ranges = np.ones(numInputs+1)
    
    if writeNNetFile is not None:
        writeNNet(weights,biases,inputMins,inputMaxes,means,ranges,writeNNetFile)
        return None
    return weights, biases


def nnet2onnx(nnetFile, onnxFile="", outputVar = "y_out", inputVar="X", normalizeNetwork=False):
    '''
    Convert a .nnet file to onnx format
    Args:
        nnetFile: (string) .nnet file to convert to onnx
        onnxFile: (string) Optional, name for the created .onnx file
        outputName: (string) Optional, name of the output variable in onnx
        normalizeNetwork: (bool) If true, adapt the network weights and biases so that 
                                 networks and inputs do not need to be normalized. Default is False.
    '''
    if normalizeNetwork:
        weights, biases = normalizeNNet(nnetFile)
    else:
        weights, biases = readNNet(nnetFile)
        
    inputSize = weights[0].shape[1]
    outputSize = weights[-1].shape[0]
    numLayers = len(weights)
    
    # Default onnx filename if none specified
    if onnxFile=="":
        onnxFile = nnetFile[:-4]+'onnx'
    
    # Initialize graph
    inputs = [helper.make_tensor_value_info(inputVar,   TensorProto.FLOAT, [inputSize])]
    outputs = [helper.make_tensor_value_info(outputVar, TensorProto.FLOAT, [outputSize])]
    operations = []
    initializers = []
    
    # Loop through each layer of the network and add operations and initializers
    for i in range(numLayers):
        
        # Use outputVar for the last layer
        outputName = "H%d"%i
        if i==numLayers-1: 
            outputName = outputVar
           
        # Weight matrix multiplication
        operations.append(helper.make_node("MatMul",["W%d"%i,inputVar],["M%d"%i]))
        initializers.append(numpy_helper.from_array(weights[i].astype(np.float32),name="W%d"%i))
            
        # Bias add 
        operations.append(helper.make_node("Add",["M%d"%i,"B%d"%i],[outputName]))
        initializers.append(numpy_helper.from_array(biases[i].astype(np.float32),name="B%d"%i))
            
        # Use Relu activation for all layers except the last layer
        if i<numLayers-1: 
            operations.append(helper.make_node("Relu",["H%d"%i],["R%d"%i]))
            inputVar = "R%d"%i
    
    # Create the graph and model in onnx
    graph_proto = helper.make_graph(operations,"nnet2onnx_Model",inputs, outputs,initializers)
    model_def = helper.make_model(graph_proto)

    # Print statements
    print("Converted NNet model at %s"%nnetFile)
    print("    to an ONNX model at %s"%onnxFile)
    
    # Additional print statements if desired
    #print("\nReadable GraphProto:\n")
    #print(helper.printable_graph(graph_proto))
    
    # Save the ONNX model
    onnx.save(model_def, onnxFile)
  

if __name__ == '__main__':
    # Read user inputs and run nnet2onnx function for different numbers of inputs
    if len(sys.argv)>1:
        nnetFile = sys.argv[1]
        if len(sys.argv)>2:
            onnxFile = sys.argv[2]
            if len(sys.argv)>3:
                outputName = argv[3]
                nnet2onnx(nnetFile,onnxFile,outputName)
            else: nnet2onnx(nnetFile,onnxFile)
        else: nnet2onnx(nnetFile)
    else:
        print("Need to specify which .nnet file to convert to ONNX!")