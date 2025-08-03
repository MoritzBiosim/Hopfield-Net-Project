# **Documentation**

A project by Moritz Wagner 2025. For private purposes only. E-Mail: wagnermojo@gmail.com

## **Content**
1. Reference
2. The Model
3. The Implementation
4. Experimental Results

## **Reference**
J.J. Hopfield, Neural networks and physical systems with emergent collective computational abilities., Proc. Natl. Acad. Sci. U.S.A. 79 (8) 2554-2558, https://doi.org/10.1073/pnas.79.8.2554 (1982).
   
## **The Model**
In 1982, J. J. Hopfield presented a mathematical framework for how nervous systems could yield content adressable memory. He demonstrated that computational abilities arise as collective properties of many simple identical units (neurons) and their interconnections. Each neuron can either be firing (1) or resting (0). The general idea is that information is encoded as a state of the system. A state is represented as an array containing the individual states of all neurons V = (V<sub>0</sub>, ... V<sub>N</sub>). For computational purposes, neurons will be converted to bipolar (-1, 1). Each neuron may update itself independently according to a mean attempt rate which results in asynchronous processing. If the weighted sum of all its inputs exceeds a certain threshold, the neuron fires (-> 1), otherwise it will stop firing. To assign some memories {S<sub>0</sub>, ..., S<sub>n</sub>}, a symmetric matrix is computed by summing the outer product over all S, thereby encoding how specific neurons "correlate" ('Neurons that fire together, wire together'). Note that the diagonal of the matrix is set to zero to avoid self-connections. This will create a dynamical system where the assigned memories are stable attractors. Given an incomplete or noisy input, the system gradually updates itself using the weight matrix, eventually retrieving a memory.

## **The Implementation**
**Please note:** 
This implementation was done irrespectively and unaware of any existing solutions or second literature, meaning this might very well not be particularly efficient or conventional. I just did it for the fun and logical exercise of building such a model.

The Model was implemented in standard python using numbpy and matplotlib. I have tried to closely follow the orignal 1982 reasoning.
For consistency, bipolar neurons were used all troughout the code. A Network is instianted trough the class *EnergyLandscape*, which takes attributes such as *numberNeurons*, *iterations*, *threshold*, etc. The heart of the code is the *asynchronousRemember* method, which emulates asynchronous parallel processing. Even when a stable state is reached, the system continues updating itself for a specified number of iterations (however, naturally its state might not change) to enable for example time sequence evolution. At the end of the code under *Exploring the Behaviour of the system* there is a collection of methods to play around with and test the model. Note that you need not give any vectors as input to the functions (even though you can, of course), as they are designed for extensive large scale testing and will randomly generate vectors of a specified number of neurons by default. The only necessary input are the desired parameters, which are usually set when executing the function. Only the matrix type is set globally. The output is generally either png-files in your directory or text written to the file 'documentation.txt'.

## **Experimental Results**
Some of the results Hopfield described are reproducible using this code. All plots were created with this implementation.

### **The Energy Function**
Hopfield defines an Energy Function which loosely speaking is a proxy for the alignment of an input with the assigned memories. The Energy function will monotonically decrease over time until a minimum is reached:
<p align="center">
  <img src="https://github.com/user-attachments/assets/8fc10315-fe9d-4718-a4ea-d64ffd2d1734" alt = "Energy decreases monotonally over time. Parameters: N = 100, assignedMemories = 5, iterations = 50, meanAttemptRate = 0.2, hamming distance of incorrect input to memory = 10" width="800">
  <br>
  <em>Energy decreases monotonally over time. Parameters: N = 100, assignedMemories = 5, iterations = 50, meanAttemptRate = 0.2, hamming distance of incorrect input to memory= 10).</em>
</p>
Even when memory recall was unsuccesful, the energy diagramm sometimes settles into a plateau. Testing reveales that this corresponds to additional stable states being reached which are not part of the assigned memories. This happens particularly when the number of assigned memories is very high, however as Hopfield pointed out, at least the exact opposite states -S are also always stable states in bipolar due to the symmetry of the system.

### **Memory Overload**
For a given number of neurons, only a certain amount of memories can be saved before error in recall is severe. In the original Hopfield paper they estimate this limit to be around 0.15N. To test this, for a given number number of assigned memories, the system simulates a specified number of runs and returns the percentage where memory recall was succesfull. We can plot the retrievability over the number of memories assigned:
<p align="center">
  <img src= "https://github.com/user-attachments/assets/ddfc67de-a7f6-4ab5-95b8-7ec1572611fc" alt = "The more memories you assign for a given number of neurons, the less reliable is recall. Parameters: N = 100, assignedMemories = 5, iterations = 50, meanAttemptRate = 0.2, hamming distance of incorrect input to memory = 10" width="800">
  <br>
  <em>The more memories you assign for a given number of neurons, the less reliable is recall. Parameters: N = 100, assignedMemories = 5, iterations = 50, meanAttemptRate = 0.2, hamming distance of incorrect input to memory = 10.</em>
</p>

### **Other Matrix Types**
Hopfield also introduced the notion of a 'clipped' matrix, meaning the sign function is called on every entry of the matrix at the end of computation (0 -> 1), yielding a matrix that only encodes the direction of correlation irrespective of strength. This is done to examine nonlinear synapses. For every expirement, 1000 random memories were selected, altered at 10% of its positions and used as input. Consistenly with Hopfield's findings, the clipped matrix yields surprisingly good results. Only if the system is close to its maximum capacity of stored memories, the performance of the clipped matrix drops sharply. <br>
Retrievability in percent over a 1000 runs (number of neurons: 100, number of mutations: 10, max iterations: 100, mean attempt rate: 0.2):<br>

<div align="center">

<table>
  <tr>
    <th></th>
    <th>default</th>
    <th>clipped</th>
  </tr>
  <tr>
    <td>5 assigned memories</td>
    <td>100.0%</td>
    <td>100.0%</td>
  </tr>
  <tr>
    <td>10 assigned memories</td>
    <td>87.9%</td>
    <td>84.7%</td>
  </tr>
  <tr>
    <td>15 assigned memories (~max capacity)</td>
    <td>90.9%</td>
    <td>13.4%</td>
  </tr>

</table>

</div>

With a random matrix, i.e. removing the symmetry by choosing random numbers between -1 and 1, Hopfield never records ergodic wandering through state space. The system seems to converge to some stable states, a simple cycle might occur, or it stays confined to a small region in state space as shown by an entropic measure. In contrary, my implementation never seems to reach any stable states with a random matrix at all. No specific reason or problem could be found so far. However the average energy and the Hamming Distance between the original input (an assigend memory) and the actual mutated version given as input, stabilize over time, indicating that here too, the system tends to become confined to a specific region within state space:
<p align="center">
  <img src= "https://github.com/user-attachments/assets/3eb2a121-6e26-417a-b658-9fbbfcbf57b2" alt = "Using a random matrix for the update rule, the average energy and the Hamming Distance to the original Input stabilize over time, maybe indicating that the systems evolution is confined to a small region of state space. Parameters: N = 100, assignedMemories = 20, iterations = 100, meanAttemptRate = 0.2, hamming distance of incorrect input to memory = 3" width="800">
  <br>
  <em>The average energy and the Hamming Distance to the original Input stabilize over time, maybe indicating that the systems evolution is confined to a small region of state space. Parameters: N = 100, assignedMemories = 20, iterations = 100, meanAttemptRate = 0.2, hamming distance of incorrect input to memory = 3.</em>
</p>

A saturated matrix, which can for example only hold values € {-3, -2, -1, 0, 1, 2, 3}, yields a provison for forgetting old memories. This can also be reproduced with this implementation, as earlier memories, indicated by a lower index, are harder to retrieve:
<p align="center">
  <img src= "https://github.com/user-attachments/assets/22a31429-11d8-4306-98d7-4119d6f9377c" alt = "The earlier a memory was added (indicated by a lower index), the harder it is to retrieve. Parameters: Number of simulations per index: 100, N = 100, assignedMemories = 20, iterations = 50, meanAttemptRate = 0.2, hamming distance of incorrect input to memory = 10" width="800">
  <br>
  <em>The earlier a memory was added (indicated by a lower index), the harder it is to retrieve. Parameters: Number of simulations per index: 100, N = 100, assignedMemories = 20, iterations = 50, meanAttemptRate = 0.2, hamming distance of incorrect input to memory = 10.</em>
</p>

Work in Progress: time sequence evolution is yet to be implemented and tested correctly.
