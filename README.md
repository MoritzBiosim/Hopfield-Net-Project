# **Documentation**

A project by Moritz Wagner 2025. E-Mail: wagnermojo@gmail.com

## **Content**
1. Reference
2. The Model
3. The Implementation
4. Experimental Results

## **Reference**
J.J. Hopfield, Neural networks and physical systems with emergent collective computational abilities., Proc. Natl. Acad. Sci. U.S.A. 79 (8) 2554-2558, https://doi.org/10.1073/pnas.79.8.2554 (1982).
   
## **The Model**
In 1982, J. J. Hopfield presented a mathematical framework for how nervous systems could yield content adressable memory. He demonstrated, that computational abilities arise as collective properties of many simple identical units (neurons) and their interconnections. Each neuron can either be firing (1) or resting (0). The general idea is that information corresponds to a state of the system. A state is represented as an array containing the individual states of all neurons V = (V<sub>0</sub>, ... V<sub>N</sub>). For computational purposes, neurons will be converted to bipolar (-1, 1). Each neuron may update itself independently according to a mean attempt rate which results in asynchronous processing. If the weighted sum of all its inputs exceeds a certain threshold, the neuron fires (-> 1), otherwise it will stop firing. To assign some memories {S<sub>0</sub>, ..., S<sub>n</sub>}, a symmetric matrix is computed by summing the outer product over all S, thereby encoding how much specific neurons "correlate". Note that the diagonal of the matrix is set to zero to avoid self-connections. This will create a dynamical system where the assigned memories are stable attractors. Given an incomplete or noisy input, the system gradually updates itself using the weight matrix, eventually retrieving a memory.

## **The Implementation**
**Please note:** 
This implementation was done irrespectively and unaware of any existing solutions or second literature, meaning this might very well not be particularly effient or conventional. Its sole purpose is the logical exercise of building such a model.

The Model was implemented in standard python using numbpy and matplotlib. I have tried to closely follow the orignal 1982 reasoning.
For consistency, bipolar neurons were used all troughout the code. A Network is instianted trough the class *EnergyLandscape*, which takes attributes such as *numberNeurons*, *iterations*, *threshold*, etc. The heart of the code is the *asynchronousRemember* method, which emulates asynchronous parallel processing. Even when a stable state is reached, the system continues updating itself for a specified number of iterations (however, naturally its state might not change) to enable for example time sequence evolution. At the end of the code under *Exploring the Behaviour of the system* there is a collection of methods to play around with and test the model.

## **Experimental Results**
Some of the results Hopfield described are reproducible using this code. All plots were created with this implementation.

Hopfield defines an Energy Function which loosely speaking is a proxy for the alignment of an input with the assigned memories. The Energy function will monotonically decrease over time until a minimum is reached:
<p align="center">
  <img src="https://github.com/user-attachments/assets/8fc10315-fe9d-4718-a4ea-d64ffd2d1734" alt = "Energy decreases monotonally over time. Parameters: N = 100, assignedMemories = 5, iterations = 50, meanAttemptRate = 0.2, hamming distance of incorrect input to memory = 10" width="800">
  <br>
  <em>Energy decreases monotonally over time. Parameters: N = 100, assignedMemories = 5, iterations = 50, meanAttemptRate = 0.2, hamming distance of incorrect input to memory= 10).</em>
</p>
Even when memory recall was unsuccesful, the energy diagramm sometimes settles into a plateau. Testing reveales that this corresponds to additional stable states being reached which are not part of the assigned memories. This happens particularly when the number of assigned memories is very high, however as Hopfield pointed out, at least the exact opposite states -S are also always stable states in bipolar due to the symmetry of the system.

For a given number of neurons, only a certain amount of memories can be saved before error in recall is severe. In the original Hopfield paper they estimate this limit to be around 0.15N. To test this, for a given number number of assigned memories, the system simulates a specified number of runs and returns the percentage where memory recall was succesfull. We can plot the retrievability over the number of memories assigned:
<p align="center">
  <img src= "https://github.com/user-attachments/assets/ddfc67de-a7f6-4ab5-95b8-7ec1572611fc" alt = "The more memories you assign for a given number of neurons, the less reliable is recall. Parameters: N = 100, assignedMemories = 5, iterations = 50, meanAttemptRate = 0.2, hamming distance of incorrect input to memory = 10" width="800">
  <br>
  <em>The more memories you assign for a given number of neurons, the less reliable is recall. Parameters: N = 100, assignedMemories = 5, iterations = 50, meanAttemptRate = 0.2, hamming distance of incorrect input to memory = 10.</em>
</p>
