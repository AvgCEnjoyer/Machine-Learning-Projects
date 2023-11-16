

class LinearRegressor:

    def __init__(self):

        self.weights = [1]
        self.data = []

        self.dim = 0

        self.learnRate = 0
        self.threshold = 0

        self.sse = 0
        

    def targetFunc(self, x):
        
        res = 0
        for i in range(self.dim):
            res += x[i] * self.weights[i]

        return res


    def gradient(self):
        
        self.sse = 0

        res = [0 for _ in range(self.dim)]

        for i in range(len(self.data)):

            funcValue = self.targetFunc(self.data[i]) 

            #res = SUM xi * (yi - f(xi))
            for j in range(self.dim):
                res[j] += self.data[i][j] * (self.data[i][-1] - funcValue)


            self.sse += (funcValue - self.data[i][-1])**2

        return res
    

    def update(self):
        
        iter = 0

        while True:
        
            sseSave = self.sse

            grad = self.gradient()
            
            outString = ""
            for i in range(len(self.weights)):
                outString += str(self.weights[i])
                if i < len(self.weights) - 1: outString += ", "
                else: outString += ","

            print(str(iter)+",", outString, str(self.sse))

            for i in range(len(self.weights)):
                self.weights[i] += self.learnRate * grad[i]

            if sseSave - self.sse < self.threshold and iter >= 1:
                break

            iter += 1
        

    def procInput(self, dataFile):

        data = []

        file = open(dataFile, "r")
        for line in file:
            dataPart = list(line.split(","))
            for i in range(len(dataPart)):
                dataPart[i] = float(dataPart[i])
            dataPartBias = [1] #x0 or bias´
            for elem in dataPart:
                dataPartBias.append(elem)
            data.append(dataPartBias)

        self.data = data
        self.dim = len(self.data[0])-1

        self.weights = [0 for _ in range(self.dim)]

        return data
    
import sys

def main():
    
    args = sys.argv

    data = ""
    eta = 0
    threshold = 0

    for i in range(len(args)):
        if args[i] == '--data':
            data = str(args[i+1])
        if args[i] == '--eta':
            eta = float(args[i+1])
        if args[i] == '--threshold':
            threshold = float(args[i+1])

    data = "LinRegRandom.csv"
    eta = 0.00005
    threshold = 0.0001

    reg = LinearRegressor()
    reg.procInput(data)
    reg.learnRate = eta
    reg.threshold = threshold

    reg.update()


if __name__ == "__main__":
    main()