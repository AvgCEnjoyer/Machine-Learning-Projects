import math
            


class DecisionTree:

    def __init__(self):

        self.entropy = None
        self.name = "root"
        self.indexes = []
        self.data = []
        self.outcomes = []

        self.base = 10

        self.depth = 0
        self.childs = []
        self.leaf = "no_leaf"

    
    def findIndex(self, list, element):
        '''
        Returns the index of an element in a list.
        '''
        for i in range(len(list)):
            if list[i] == element: return i
        return None
    

    def overrideAttribAmount(self, data, outcome):
        '''
        Gets a partial list of a getValueAmount list and adds 1
        to the index of the given targetAttribute value.
        '''
        if outcome not in self.outcomes: 
            print("something went wrong")
            raise Exception(data, outcome, self.outcomes)
        
        index = -1
        for i in range(len(self.outcomes)):
            if outcome == self.outcomes[i]:
                index = i

        data[index] += 1


    def getValues(self, index):
        '''
        Gets all values that are in the datatable at
        place index.
        '''
        valueSet = []

        for element in self.data:
            value = element[index]
            if value not in valueSet:
                valueSet.append(value)

        return valueSet
    

    def getValueAmount(self, index):
        '''
        Returns the values of the data set with the amount
        of values in form:

        [ value0, amount0, ..., valueN, amountN]
        '''
        valueSet = []

        for element in self.data:
            value = element[index]
            if value not in valueSet:
                valueSet.append(value)
                valueSet.append(1)
            else:
                valueSet[self.findIndex(valueSet, value)+1] += 1


        return valueSet

    
    def getValueAmountPartial(self, index):
        '''
        Returns the values of the data set with the amount
        of outcomes in form:

        [ value0, [ |outcome0|, ..., |outcomeN| ], ... ]
        '''
        valueSet = []

        for element in self.data:

            value = element[index]

            #Check if value is not in the result list
            if value not in valueSet:

                #Set up initial amounts for this value
                valueSet.append(value)
                valueSet.append([0 for _ in range(len(self.outcomes))])

                #Set up index to be the last item, which is the one we just added
                indx = -1

            else:
                indx = self.findIndex(valueSet, value) + 1

            #Increase the amount of the given value (+1)
            self.overrideAttribAmount(valueSet[indx], element[-1])

        return valueSet
    

    def getValueSum(self, data):
        '''
        Computes the values of a list at every other index.
        Output with list[a1,a2,...,an] : a2+a4+a6...
        '''
        sum = 0
        for i in range(1,len(data),2):
            sum += data[i]

        return sum


    def calculateEntropy(self, index):
        '''
        Computes the general entropy of the node
        '''
        
        # Get the amount of different values at place index in teh datatable
        values = self.getValueAmount(index)
        # values is of form : [attrib0, value0, ..., attribN, valueN]]

        entropySum = 0

        # Get the values at every other place in values, no need to consider teh attributeName
        sumValues = self.getValueSum(values)
        
        #Calculate entropy by the given formula
        for i in range(1, len(values), 2):
            fraction = values[i] / sumValues
            entropySum -= fraction * math.log(fraction, self.base)
        
        return entropySum
    

    def calculateEntropyPartial(self, index):
        '''
        Calculates the partial entropies of each value at place index 
        of a given attribute in the datatable. Returns the weighted sum 
        of all partial entropies
        '''
        # Get the amount of classification outcomes for each attributeValue
        values = self.getValueAmountPartial(index)
        #Values is of form : [value0, [ |outcome0|, ..., |outcomeN| ], ..., attribN, [ |outcome0|, ..., |outcomeN| ]]

        #Calculate Weighted sum of entropies
        entropySum = 0.0

        #Go through values and only take every other element with the corresponding outcome values/amounts
        for i in range(1, len(values), 2):

            partialEntropy = 0.0

            #Process each outcome with the entropie formula and the coresponding fraction 
            for j in range(len(self.outcomes)):

                pi = values[i][j] / sum(values[i])
                #Id pi is 0, this partial entropy logically is also 0, no need to compute
                if pi == 0: partialEntropy -= 0.0

                else: 
                    #Take the base of the root node for this computation
                    partialEntropy -= pi * math.log(pi, self.base)

            entropySum += ( sum(values[i]) / len(self.data) ) * partialEntropy

        return entropySum
    

    def filterList(self, data, index):
        '''
        Remove all entries at place index in the
        list "data".
        '''
        res = [[] for _ in range(len(data))]

        for i in range(len(data)):
            for j in range(len(data[i])):
                if j != index:
                    res[i].append(data[i][j])
        return res
    

    def takeMajorityClass(self):
        '''
        Compute the highest amount of classification values
        with respect to the remaining attribute values. 
        '''
        #Get last possible amounts of different values
        values = self.getValueAmount(-1)
        maxValue = ["", 0]

        #Simple get-max loop
        for i in range(1, len(values), 2):
            if values[i] > maxValue[1]:
                maxValue = [values[i-1], values[i]]

        return maxValue[0]
    

    def update(self):
        
        #0 No more classes to split upon? Take the majority class and Entropy locically is 0 then.
        if len(self.data) <= 1 or len(self.data[0]) <= 1:
            self.entropy = 0.0
            self.leaf = self.takeMajorityClass()
            self.printTree()
            return
        
        #1 : calculate Entropy
        self.entropy = self.calculateEntropy(-1)

        #2 : If Entropy = 0 : END
        if self.entropy == 0.0: 
            self.leaf = self.data[0][-1]
            self.printTree()
            return

        #3 : Compute entropies of attributes
        attribEntropies = []
        for i in range(len(self.data[0])-1): #-1 because we already computed the last attribute's entropy
            attribEntropies.append(self.calculateEntropyPartial(i))

        #4 : calculate Inforamation Gain
        gains = [0 for _ in range(len(attribEntropies))]

        for i in range(len(gains)):
            gains[i] = self.entropy - attribEntropies[i]

        #5 : Pick Attribute with highest Infogain as next node
        maxIndex = 0
        for i in range(len(gains)):
            if gains[i] > gains[maxIndex]:
                maxIndex = i

        #6 : Create Child nodes for the outcomes of the best follow up atrribute

        #->Print the current node before computing the childs
        self.printTree()

        # Get the different values for the chosen attribute to create branches for each of them
        values = self.getValues(maxIndex)

        # Reverse values because the solution car.csv says so, normally the output would be computed 
        # in a way that the first appearance of a value would get the first child branch.
        values.reverse()

        #Create child nodes
        for i in range(len(values)):
            '''
            - Name of the child is predefined in the submission text as attribN=Value
            - Each child gets the base of the root
            - Each child gets a filtered list which doesn't contain the currently chosen attribute value
            '''
            child = DecisionTree()
            child.name = "att"+str(self.indexes[maxIndex])+"="+str(values[i])
            child.base = self.base

            #Calculate remaining attribute names
            indx = [self.indexes[i] for i in range(len(self.indexes)) if i != maxIndex]

            #Set child depth
            child.depth = self.depth + 1
            
            #Cleanse chosen attribute value from the list
            data = list(filter(lambda x : x[maxIndex] == values[i], self.data))

            #Set up child Data
            child.loadData(self.filterList(data, maxIndex), indx)
            child.update()

    
    def printTree(self):
        
        print(self.depth, ",", self.name,",", self.entropy,",", self.leaf)


    def loadData(self, data, indexes):
        
        self.indexes = indexes
        self.data = data

        targetValues = []
        for element in self.data:
            if element[-1]  not in targetValues: 
                targetValues.append(element[-1])

        self.outcomes = targetValues
            

    def loadFile(self, dataFile):

        file = open(dataFile, "r")
        for line in file:
            self.data.append(list(line.split(",")))

        for element in self.data:
            for i in range(len(element)):
                if type(element[i]) == str:
                    element[i] = element[i].replace("\n", "")

        #---Edgy data cleansing (Test Data had an element ["] that need to be filtered out)
        for element in self.data:
            if len(element) != len(self.data[0]):
                self.data.remove(element)
        #----------------------------------------------------------------------------------

        targetValues = []
        for element in self.data:
            if element[-1]  not in targetValues: 
                targetValues.append(element[-1])

        self.indexes = [i for i in range(len(self.data[0]))]
        self.outcomes = targetValues
        self.base = len(targetValues)


if __name__ == "__main__":

    import sys

    data = ""

    args = sys.argv

    for i in range(len(args)):
        if args[i] == '--data':
                data = str(args[i+1])
    
    data = "DecisionTreeCarData.csv"

    tree = DecisionTree()
    tree.loadFile(data)

    tree.update()
