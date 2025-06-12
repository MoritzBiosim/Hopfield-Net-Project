
"""Recreating the 1982 Hopfield Model using Numbpy. All neurons are bipolar. 
Results are written to documentation.txt. Project by Moritz Wagner 2025"""

import numpy as np
import matplotlib.pyplot as plt

### Energy Landscape Class ###

class EnergyLandscape():
    """Each object is a unique energy landscape formed by the states 
    which were used to compute the weight matrix. Each state thus becomes a stable attractor, 
    i.e. a local minimum in the landscape"""

    def __init__(self, numberNeurons, states, iterations, meanAttemptRate, threshold = 0):
    
        self.numberNeurons = numberNeurons
        self.states = states
        self.iterations = iterations
        self.meanAttemptRate = meanAttemptRate
        self.threshold = threshold

    def getStates(self):
        """Returns the list of states"""
        return self.states
    
    def addState(self, state):
        """Adds a given state to the list of states"""
        self.states.append(state)
        return self.states
    
    def createWeightMatrix(self):
        """Creates a weight matrix from the given states by summing the outer product 
        of all states with themselves. The resulting matrix 
        is symmetric and its diagonal is set to zero."""
        
        matrix = np.zeros((self.numberNeurons, self.numberNeurons))
        for state in self.states:
            matrix += np.outer(state, state)
        "Tii = 0:"
        np.fill_diagonal(matrix, 0)
        
        return matrix   
    
    def createClippedMatrix(self):
        "Each element in the matrix greater than or equal to 0 is set to 1, otherwise -1. The diagonal is set to 0."
        matrix = np.zeros((self.numberNeurons, self.numberNeurons))
        for state in self.states:
            matrix += np.outer(state, state)
        clippedMatrix = np.where(matrix >= 0, 1, -1) #maybe sign function would also be adequate here?
        np.fill_diagonal(clippedMatrix, 0)

        return clippedMatrix
    
    def createSaturatedMatrix(self):
        """The elements of the matrix can only take the values {-3, -2, -1, 0, 1, 2, 3}. 
        This leads to older memories automatically being forgotten."""
        saturatedMatrix = np.zeros((self.numberNeurons, self.numberNeurons))
        for state in self.states:
            update = np.outer(state, state)
            mask = (saturatedMatrix != 3) & (saturatedMatrix != -3)
            saturatedMatrix[mask] += update[mask]
        np.fill_diagonal(saturatedMatrix, 0)

        return saturatedMatrix

    def createRandomMatrix(self):
        """Generates a random weight matrix with given dimensions. Values are any number between -1 and 1, the diagonal is set to 0."""
        #matrix = np.random.choice([-1, 1], size=(self.numberNeurons, self.numberNeurons))
        #matrix = np.random.normal(loc = 0.0, scale = np.sqrt(1/self.numberNeurons), size=(self.numberNeurons, self.numberNeurons))
        matrix = np.random.uniform(-1., 1., size=(self.numberNeurons, self.numberNeurons))
        np.fill_diagonal(matrix, 0)

        return matrix


    def checkForStability(self, state, matrix):
        "If an attractor is reached, the state will remain unchanged after transformation. The function returns a boolean."
        copy = state.copy()
        weightedInputs = np.dot(matrix, copy)   #100 stellen vector
        transformedState = np.where(weightedInputs >= self.threshold,  1, -1)   #größer gleich
        stablility = np.array_equal(state, transformedState)
        return stablility
    

    def asynchronousRemember(self, matrix, input, originalInput):
        """Given an incomplete or noisy input, this method updates the input vector
        using the weight matrix until a local minimum in the energy landscape is reached. 
        This is achieved by asynchronous processing, meaning for a fixed number of iterations, 
        each neuron gets a chance to update itself independently according to the meanAttemptRate. 
        If so, the weighted sum of all its inputs is computed and compared to the threshold.
        If the weighted sum exceeds the threshold, the neuron is set to 1, otherwise to -1. 
        The function terminates when the maximum number of iterations is reached. 
        It returns a tuple containing a set of all attractors reached, 
        a set containing all memories recalled during the process (eg. to enable 
        time sequence evolution tracking further down the line), the energy values 
        over each iteration and the Hamming distance to the originalInput (input before mutation) over time."""

        #documentation.write(f"matrix: {matrix}\n")
        numUpdates = 0
        energyTracker = []
        attractor = set()
        memories = set()
        hammingDistance = []
        #mutatedInput = input.copy()

        for i in range(self.iterations):                                     
            "in each iteration, every neuron gets a chance to update itself according "
            "to the meanAttemptRate"
            #documentation.write(f"--Iteration {i+1}--\n")
            energy = calculateEnergy(matrix, input)
            energyTracker.append(energy)
            hammingDistance.append(getHammingDistance(input, originalInput))
            energy = 0 #redundant?
            "Check if the state has converged to a memory"

            for  index, neuron in enumerate(input):
                "random chance for each neuron to update during iteration"
                if np.random.rand() < self.meanAttemptRate:
                    "compute the weighted sum of all inputs"
                    weightedInputs = np.dot(matrix[index], input)
                    #documentation.write(f"Neuron {index} updated\n")
                    numUpdates += 1
                    "set neuron to 1 or 0 depending on  the threshold and the weighted sum"
                    if weightedInputs >= self.threshold: #größer gleich?
                        input[index] = 1 
                    else: 
                        input[index] = -1 

            stability = self.checkForStability(input, matrix)

            if stability:
                attractor.add(str(input))
                for s, state in enumerate(self.states):
                    if np.array_equal(input, state) == True:
                        #documentation.write(f"Stable state number {s} reached after {i} iterations and {numUpdates} updates.\n")       #f"Stable state number {s} reached: {state}, iterations, {i+1}, transformed input: {input}, Input before transformation: {mutatedInput}, updates: {numUpdates}\n"
                        memories.add(str(state)) #elegantere Lösung nötig?
        
        return (attractor, memories, energyTracker, hammingDistance) #returns the latest state and the energy landscape



### General Functionalities ###

def generateStates(numberStates, numberNeurons):
    """Create a list of states with random neurons"""
    
    states = []
    for i in range(numberStates):
        state = np.array([np.random.choice([-1, 1]) for i in range(numberNeurons)]) #Binary or bipolar?
        states.append(state)
        
    return states


def mutateState(input, numberNeurons, numberMutations=1):
    """Mutates the state by flipping a given number of neurons at random positions"""
       
    for i in range(numberMutations):
        index = np.random.randint(0, numberNeurons-1)
        input[index] = -input[index]
        #documentation.write(f"index mutated {index}\n")
    return input


def choseInput(states, numberNeurons, numberStates, numberMutations):
    """Randomly chooses a state from the list of memorized states and mutates it 
    by a given number of mutations. Returns both the mutated and the original input."""
    inputIndex = np.random.randint(numberStates)
    originalInput = states[inputIndex].copy()
    #documentation.write(f"State number {inputIndex} is chosen from all assigned memories as input: {originalInput}\n")
    mutatedInput = mutateState(originalInput.copy(), numberNeurons, numberMutations)
    #documentation.write(f"After altering {numberMutations} positions, the now incorrect input is: {mutatedInput}\n")
    return mutatedInput, originalInput


def calculateEnergy(matrix, state):
    """Calculates the energy of the current state using NumPy."""
    return -0.5 * np.dot(state, np.dot(matrix, state))

def getMeanEnergyOverTime(energyTracker):
    """Calculates the mean energy over time from the energy tracker."""
    #maybe use numbpy's cumsum to improve efficiency as suggested by copilot
    sumEnergy = 0
    meanEnergyOverTime = np.zeros(len(energyTracker))
    for i in range(0,len(energyTracker)):
        sumEnergy += energyTracker[i]
        meanEnergy = sumEnergy/(i+1)
        meanEnergyOverTime[i] = meanEnergy
    return meanEnergyOverTime

def getHammingDistance(state1, state2):
    ""
    if len(state1) != len(state2): raise ValueError("States must be of equal number neurons to compute Hamming Distance!")
    differences = (state1 != state2)
    hammingDistance = np.count_nonzero(differences)
    return hammingDistance


def plotEnergy(energyTracker):
    """Plots the energy landscape for a run"""
    x = range(0, len(energyTracker))
    plt.figure(figsize=(10, 5))
    plt.plot(x, energyTracker, label = 'Energy over time', color = 'blue', linestyle = 'dotted')
    plt.plot(x, getMeanEnergyOverTime(energyTracker), label = 'Mean energy over time', color = 'green', linestyle = 'dotted')
    step = max(1, len(energyTracker) // 10)
    plt.xticks(range(0, len(energyTracker), step))
    plt.xlabel('Iterations')
    plt.ylabel('Energy')
    plt.title('Energy Landscape')
    plt.legend()
    plt.tight_layout()
    plt.savefig('energy_landscape.png')
    plt.close()


### Exploring the Behaviour of the system ### 

def plotEnergyfunction(numberStates, numberNeurons, numberMutations, iterations, meanAttemptRate, matrixType, threshold = 0, states = None, input = None):
    "Plots the energy function for a single run"
    if type(numberStates) != int:
        raise TypeError("Number of States must be a single int for this function")
    
    if not states:
        states = generateStates(numberStates, numberNeurons)

    if not input:
        input, originalInput = choseInput(states, numberNeurons, numberStates, numberMutations)
    else:
        originalInput = input.copy()

    energyLandscape = EnergyLandscape(numberNeurons, states, iterations, meanAttemptRate, threshold)    

    if matrixType == "default": 
        matrix = energyLandscape.createWeightMatrix()
    elif matrixType == "clipped":
        matrix = energyLandscape.createClippedMatrix()
    elif matrixType == "saturated":
        matrix = energyLandscape.createSaturatedMatrix()
    elif matrixType == "random":
        matrix = energyLandscape.createRandomMatrix()
    else:
        raise ValueError("Matrix type must be 'default' or 'clipped' or 'saturated' or 'random'.")
    
    result = energyLandscape.asynchronousRemember(matrix, input, originalInput)
    plotEnergy(result[2])
    documentation.write(f"attractors reached: {len(result[0])}, designated memories recalled: {len(result[1])}\n")


def getRetrievability(numberRuns, numberStates, numberNeurons, numberMutations, iterations, meanAttemptRate, matrixType, threshold = 0, states = None):
    """The function creates a weigth matrix from random states if not specified differently 
    and for a given number of runs chooses a random nominal memory, 
    mutating it by a given number of digits and tries to recollect the now incorrect memory. 
    The function returns the number of successful recollections and the average number of iterations 
    it took to retrieve the memory for any given number of assigned memories and neurons."""
    documentation.write(f"---getRetrievability---\n")
    documentation.write(f"Number of runs: {numberRuns}, number of states: {numberStates}, number of neurons: {numberNeurons}, number of mutations: {numberMutations}, max iterations: {iterations}, mean attempt rate: {meanAttemptRate}, matrix type: {matrixType}\n")
   
    retrievabilityCount = 0
    stabilisationCount = 0

    if not states:
        states = generateStates(numberStates, numberNeurons)
    
    energyLandscape = EnergyLandscape(numberNeurons, states, iterations, meanAttemptRate, threshold)
    
    if matrixType == "default": 
        matrix = energyLandscape.createWeightMatrix()
    elif matrixType == "clipped":
        matrix = energyLandscape.createClippedMatrix()
    elif matrixType == "saturated":
        matrix = energyLandscape.createSaturatedMatrix()
    elif matrixType == "random":
        matrix = energyLandscape.createRandomMatrix()
    else:
        raise ValueError("Matrix type must be 'default' or 'clipped' or 'saturated' or 'random'.")
    
    for r in range(numberRuns):
        
        attractorReached = False
        memoryRetrieved = False
        input, originalInput = choseInput(states, numberNeurons, numberStates, numberMutations)
        result = energyLandscape.asynchronousRemember(matrix, input, originalInput)
        if len(result[0]) > 0:
            attractorReached = True
        if len(result[1]) > 0:
            memoryRetrieved = True
        retrievabilityCount += memoryRetrieved
        stabilisationCount += attractorReached
        relativeRetrievability = retrievabilityCount / numberRuns *100
        relativeSabilisation = stabilisationCount / numberRuns *100

    documentation.write(f"A memory was retrieved {retrievabilityCount} times out of {numberRuns} Runs. That equals to {relativeRetrievability}%. Any attractor was reached in {relativeSabilisation}% of Runs.\n")
    return relativeRetrievability


def plotRetrievabilityOverNumberStates(numberRuns, numberStates, numberNeurons, numberMutations, iterations, meanAttemptRate, matrixType):
    """The function takes a list containing different numbers of states memorized
      for a given amount of neurons and plots the retrievability in percent 
      (how many inputs are recognized and remembered correctly). It is possible 
      to demonstrate the faltering of recollection with incereasing number of states per neuron."""
    if type(numberStates) != list:
        raise TypeError("Number of States must be a list for this function")
    retrievability = []
    for sim in range(len(numberStates)):
        retrievability.append(getRetrievability(numberRuns, numberStates[sim], numberNeurons, numberMutations, iterations, meanAttemptRate, matrixType))
    plt.figure(figsize=(10, 5))
    plt.plot(numberStates, retrievability)
    plt.xlabel('Number of States')
    plt.ylabel('Retrievability (%)')
    plt.title('Retrievability vs Number of States')
    plt.tight_layout()
    plt.savefig('retrievability_vs_number_of_states.png')
    plt.close()


####### PLAYGROUND #######
print("thinking...")

matrixType = "saturated"  #"default", "clipped", "saturated", "random" 

with open("documentation.txt", "a") as documentation:
    documentation.write(f"---HOPFIELD NET DOCUMENTATION---\n")
    #plotRetrievabilityOverNumberStates(numberRuns = 10, numberStates = [1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100], numberNeurons = 100, numberMutations = 10, iterations = 50, meanAttemptRate = 0.2, matrixType = matrixType)
    plotEnergyfunction(numberStates = 20, numberNeurons = 100, numberMutations = 5, iterations = 300, meanAttemptRate = 0.2, matrixType = matrixType)
    #getRetrievability(numberRuns = 100, numberStates = 20, numberNeurons = 100, numberMutations = 10, iterations = 50, meanAttemptRate = 0.2, matrixType = matrixType)


### TO DO ###

#np.functions mit Formeln aus dem Paper vergleichen (sollte stimmen)

### Beobachtungen ###

### Future Work ###

#saturation of size of Tij probieren: Tij E {0, +-1, +-2, +-3}
#time sequence evolution probieren (vielleicht anders als bei Hopfield zusammenhängende 
# Erinnerungen codieren für einander, also sequence evolution nicht in connections veranlagt?)

