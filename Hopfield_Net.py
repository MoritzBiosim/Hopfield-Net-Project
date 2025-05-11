
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
    
    def createWeigthMatrix(self):
        """Creates a weight matrix from the given states by summing the outer product 
        of all states with themselves. The resulting matrix 
        is symmetric and its diagonal is set to zero."""
        
        matrix = np.zeros((self.numberNeurons, self.numberNeurons))
        for state in self.states:
            matrix += np.outer(state, state)
        "Tii = 0:"
        np.fill_diagonal(matrix, 0)
        
        return matrix
    
    def generateRandomMatrix(self):
        """Generates a random weight matrix with given dimensions. Values are either -1 or 1, the diagonal is set to 0."""

        matrix = np.random.uniform(-1, 1, size=(self.numberNeurons, self.numberNeurons))
        np.fill_diagonal(matrix, 0)

        return matrix


    def checkForStability(self, state, matrix):
        "If an attractor is reached, the state will remain unchanged after transformation. The function returns a boolean."
        copy = state.copy()
        weightedInputs = np.dot(matrix, copy)
        transformedState = np.where(weightedInputs >= self.threshold,  1, -1)   #größer gleich
        stablility = np.array_equal(state, transformedState)
        return stablility
    

    def asynchronousRemember(self, matrix, input):
        """Given an incomplete or noisy input, this method updates the input vector
        using the weight matrix until a local minimum in the energy landscape is reached. 
        This is achieved by asynchronous processing, meaning for a fixed number of iterations, 
        each neuron gets a chance to update itself independently according to the meanAttemptRate. 
        If so, the weighted sum of all its inputs is computed and compared to the threshold.
        If the weighted sum exceeds the threshold, the neuron is set to 1, otherwise to -1. 
        The function terminates when the maximum number of iterations is reached. 
        It returns a tuple containing a boolean indicating if a stable state was reached, 
        a set containing all memories recalled during the process (eg. to enable 
        time sequence evolution tracking further down the line), and the energy values 
        over each iteration."""

        #documentation.write(f"matrix: {matrix}\n")
        numUpdates = 0
        energyTracker = []
        attractor = False
        memories = set()
        #mutatedInput = input.copy()

        for i in range(self.iterations):                                     
            "in each iteration, every neuron gets a chance to update itself according "
            "to the meanAttemptRate"
            #documentation.write(f"--Iteration {i+1}--\n")
            energy = calculateEnergy(matrix, input)
            energyTracker.append(energy)
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
                attractor = True
                for s, state in enumerate(self.states):
                    if np.array_equal(input, state) == True:
                        #documentation.write(f"Stable state number {s} reached after {i} iterations and {numUpdates} updates.\n")       #f"Stable state number {s} reached: {state}, iterations, {i+1}, transformed input: {input}, Input before transformation: {mutatedInput}, updates: {numUpdates}\n"
                        memories.add(str(state)) #elegantere Lösung nötig?
        
        return (attractor, memories, energyTracker) #returns the latest state and the energy landscape



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
    by a given number of mutations"""
    inputIndex = np.random.randint(numberStates)
    originalInput = states[inputIndex].copy()
    #documentation.write(f"State number {inputIndex} is chosen from all assigned memories as input: {originalInput}\n")
    mutatedInput = mutateState(originalInput, numberNeurons, numberMutations)
    #documentation.write(f"After altering {numberMutations} positions, the now incorrect input is: {mutatedInput}\n")
    return mutatedInput


def calculateEnergy(matrix, state):
    """Calculates the energy of the current state using NumPy."""
    return -0.5 * np.dot(state, np.dot(matrix, state))


def plotEnergy(energyTracker):
    """Plots the energy landscape for a run"""
    x = range(0, len(energyTracker))
    plt.figure(figsize=(10, 5))
    plt.plot(x, energyTracker)
    step = max(1, len(energyTracker) // 10)
    plt.xticks(range(0, len(energyTracker), step))
    plt.xlabel('Iterations')
    plt.ylabel('Energy')
    plt.title('Energy Landscape')
    plt.tight_layout()
    plt.savefig('energy_landscape.png')
    plt.close()


def getRetrievability(numberRuns, numberStates, numberNeurons, numberMutations, iterations, meanAttemptRate, randomMatrix, threshold = 0, states = None):
    """The function creates a weigth matrix from random states if not specified differently 
    and for a given number of runs chooses a random nominal memory, 
    mutating it by a given number of digits and tries to recollect the now incorrect memory. 
    The function returns the number of successful recollections and the average number of iterations 
    it took to retrieve the memory for any given number of assigned memories and neurons."""
    documentation.write(f"---getRetrievability---\n")
    documentation.write(f"Number of runs: {numberRuns}, number of states: {numberStates}, number of neurons: {numberNeurons}, number of mutations: {numberMutations}, max iterations: {iterations}, mean attempt rate: {meanAttemptRate}, random matrix: {randomMatrix}\n")
   
    retrievabilityCount = 0
    stabilisationCount = 0

    if not states:
        states = generateStates(numberStates, numberNeurons)
    
    energyLandscape = EnergyLandscape(numberNeurons, states, iterations, meanAttemptRate, threshold)
    
    if not randomMatrix: 
        matrix = energyLandscape.createWeigthMatrix()
    else:
        matrix = energyLandscape.generateRandomMatrix()
    
    for r in range(numberRuns):
        
        memoryRetrieved = False
        input = choseInput(states, numberNeurons, numberStates, numberMutations)
        result = energyLandscape.asynchronousRemember(matrix, input)
        if len(result[1]) > 0:
            memoryRetrieved = True
        retrievabilityCount += memoryRetrieved
        relativeRetrievability = retrievabilityCount / numberRuns *100
        stabilisationCount += result[0]
        relativeSabilisation = stabilisationCount / numberRuns *100

    documentation.write(f"A memory was retrieved {retrievabilityCount} times out of {numberRuns} Runs. That equals to {relativeRetrievability}%.\n")
    return relativeRetrievability


### Exploring the Behaviour of the system ### 

def plotEnergyfunction(numberStates, numberNeurons, numberMutations, iterations, meanAttemptRate, randomMatrix, threshold = 0, states = None, input = None):
    "Plots the energy function for a single run"
    if type(numberStates) != int:
        raise TypeError("Number of States must be a single int for this function")
    
    if not states:
        states = generateStates(numberStates, numberNeurons)

    if not input:
        input = choseInput(states, numberNeurons, numberStates, numberMutations)

    energyLandscape = EnergyLandscape(numberNeurons, states, iterations, meanAttemptRate, threshold)    

    if not randomMatrix: 
        matrix = energyLandscape.createWeigthMatrix()
    else:
        matrix = energyLandscape.generateRandomMatrix()

    energy = energyLandscape.asynchronousRemember(matrix, input)
    plotEnergy(energy[2])
    documentation.write(f"attractor reached: {energy[0]}, designated memories recalled: {len(energy[1])}")


def plotRetrievabilityOverNumberStates(numberRuns, numberStates, numberNeurons, numberMutations, iterations, meanAttemptRate, randomMatrix):
    """The function takes a list containing different numbers of states memorized
      for a given amount of neurons and plots the retrievability in percent 
      (how many inputs are recognized and remembered correctly). It is possible 
      to demonstrate the faltering of recollection with incereasing number of states per neuron."""
    if type(numberStates) != list:
        raise TypeError("Number of States must be a list for this function")
    retrievability = []
    for sim in range(len(numberStates)):
        retrievability.append(getRetrievability(numberRuns, numberStates[sim], numberNeurons, numberMutations, iterations, meanAttemptRate, randomMatrix))
    plt.figure(figsize=(10, 5))
    plt.plot(numberStates, retrievability)
    plt.xlabel('Number of States')
    plt.ylabel('Retrievability (%)')
    plt.title('Retrievability vs Number of States')
    plt.tight_layout()
    plt.savefig('retrievability_vs_number_of_states.png')
    plt.close()


####### PLAYGROUND #######


with open("documentation.txt", "w") as documentation:
    documentation.write(f"---HOPFIELD NET DOCUMENTATION---\n")
    plotRetrievabilityOverNumberStates(numberRuns = 10, numberStates = [1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100], numberNeurons = 100, numberMutations = 10, iterations = 50, meanAttemptRate = 0.2, randomMatrix = False)
    plotEnergyfunction(numberStates = 50, numberNeurons = 100, numberMutations = 10, iterations = 50, meanAttemptRate = 0.2, randomMatrix = False)


#Energy functon kontrollieren (seems to work)
#asynchronousRemember kontrollieren (seems to work)

#für N/2 States gespeichert bei 100 Iterations und mind 10 Fehlern konvergiert die energy und erreicht ein Plateau -> anscheinend weil stabile Zustände erreicht werden, die nicht teil der assigned memories sind!

#clipped matrix probieren: Tij = sign(Tij)
#random Matrix funktioniert probieren (funktioniert bisher nicht)
#saturation of size of Tij probieren: Tij E {0, +-1, +-2, +-3}
#time sequence evolution probieren

