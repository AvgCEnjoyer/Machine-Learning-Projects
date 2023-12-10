import math

class Perceptron:

    def __init__(self):

        self.weights : list = []
        self.inputs : list = []

        self.output : float = 0.0

        self.error = 0.0

    def calculateNet(self, inputs : list):

        netSum : float = 0.0

        for i in range(len(self.weights)):
            netSum += self.weights[i] * inputs[i]

        return netSum


    def getActivation(self, inputs : list):
        
        self.inputs = inputs
        sigmoid = 1 / (1+math.exp(-self.calculateNet(inputs)))
        self.output = sigmoid

        return sigmoid
    

    def applyRound(self):

        for i in range(len(self.weights)):
            self.weights[i] = round(self.weights[i], 5)
    

    def applyDelta(self, expectedOut, learningRate, hidden = True, delta_ = 0.0, weight = 0.0):

        if hidden:
            delta = self.output * (1 - self.output) * delta_ * weight
            self.error = delta
            for i in range(len(self.weights)):
                self.weights[i] += learningRate * delta * self.inputs[i]

            #self.applyRound()
            return

        #For output unit

        delta = (expectedOut - self.output) * self.output * (1-self.output)
        self.error = delta

        for i in range(len(self.weights)):
            
            res = learningRate * delta * self.inputs[i]

            self.weights[i] += res

        #self.applyRound()
        return delta


class NeuralNetwork:

    def __init__(self):

        self.inputs : list = []

        self.perceptrons : list = []
        self.outPerceptron = 0

        self.data : float = []
        self.expectedOut : float = 0.0

        self.learningRate = 0.0

        self.perceptrons = []
        self.outPerceptron = None

        self.__setup()

    
    def __setup(self):

        h1 = Perceptron()
        h1.weights = [0.20000, -0.30000,  0.40000]

        h2 = Perceptron()
        h2.weights = [-0.50000, -0.10000, -0.40000]

        h3 = Perceptron()
        h3.weights = [0.30000, 0.20000, 0.10000]

        out = Perceptron()
        out.weights = [-0.10000, 0.10000, 0.30000, -0.40000]

        self.perceptrons = [h1, h2, h3]
        self.outPerceptron = out


    def update(self):

        h1 = self.perceptrons[0]
        h2 = self.perceptrons[1]
        h3 = self.perceptrons[2]

        for i in range(len(self.data)):

            self.inputs = [1.00000, self.data[i][0], self.data[i][1]]
            self.expectedOut = self.data[i][-1]
            weights = self.outPerceptron.weights.copy()

            outputsLayer1 = [1.00000] #Bias

            for perceptron in self.perceptrons:
                outputsLayer1.append(perceptron.getActivation(self.inputs))

            output = self.outPerceptron.getActivation(outputsLayer1)
            
            delta = self.outPerceptron.applyDelta(self.data[i][-1], self.learningRate, hidden = False)
            for j, perceptron in enumerate(self.perceptrons):
                perceptron.applyDelta(output, self.learningRate, True, delta, weights[j+1])

            # to print:
            # a,b,h1,h2,h3,o,t,delta_h1,delta_h2,delta_h3,delta_o,w_bias_h1,w_a_h1,w_b_h1,w_bias_h2,w_a_h2,w_b_h2,w_bias_h3,w_a_h3,w_b_h3,w_bias_o,w_h1_o,w_h2_o,w_h3_o

            '''
            print(self.data[i][0], self.data[i][1], h1.output, h2.output, h3.output, output, self.data[i][-1],
                  h1.error, h2.error, h3.error, self.outPerceptron.error, 
                  h1.weights[0], h1.weights[1], h1.weights[2],
                  h2.weights[0], h2.weights[1], h2.weights[2],
                  h3.weights[0], h3.weights[1], h3.weights[2],
                  self.outPerceptron.weights[0], self.outPerceptron.weights[1], self.outPerceptron.weights[2], self.outPerceptron.weights[3])
            '''

            print(self.data[i][0], ",", self.data[i][1], ",", h1.output, ",", h2.output, ",", h3.output, ",", output, ",", self.data[i][-1], ",", h1.error, ",", h2.error, ",", h3.error, ",", self.outPerceptron.error, ",", h1.weights[0], ",", h1.weights[1], ",", h1.weights[2], ",", h2.weights[0], ",", h2.weights[1], ",", h2.weights[2], ",", h3.weights[0], ",", h3.weights[1], ",", h3.weights[2], ",", self.outPerceptron.weights[0], ",", self.outPerceptron.weights[1], ",", self.outPerceptron.weights[2], ",", self.outPerceptron.weights[3])


    def start(self, iterations):

        for _ in range(iterations):
            self.update()


    def loadFile(self, path):

        file = open(path, "r")

        content = []

        for line in file:
            tempContent = []
            for element in line.split(","):
                tempContent.append(float(element))
            content.append(tempContent)

        self.data = content

    
if __name__ == "__main__":

    import sys

    data = ""
    eta = 0.0
    iterations = 0

    args = sys.argv

    for i in range(len(args)):
        if args[i] == '--data':
                data = str(args[i+1])
        if args[i] == '--eta':
                eta = float(args[i+1])
        if args[i] == '--iterations':
                iterations = int(args[i+1])
    
    #data = "Gauss3.csv"
    #eta = 0.2
    #iterations = 1
     
    print("a,b,h1,h2,h3,o,t,delta_h1,delta_h2,delta_h3,delta_o,w_bias_h1,w_a_h1,w_b_h1,w_bias_h2,w_a_h2,w_b_h2,w_bias_h3,w_a_h3,w_b_h3,w_bias_o,w_h1_o,w_h2_o,w_h3_o")
    print("-,-,-,-,-,-,-,-,-,-,-,   0.20000,  -0.30000,   0.40000,  -0.50000,  -0.10000,  -0.40000,   0.30000,   0.20000,   0.10000,  -0.10000,   0.10000,   0.30000,  -0.40000")
    net = NeuralNetwork()
    net.loadFile(data)
    net.learningRate = eta

    net.start(iterations)