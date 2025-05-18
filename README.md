# **Documentation**

## **Content**
1. Reference
2. The Model
3. The Implementation
4. Experimental Results

## **Reference**
J.J. Hopfield, Neural networks and physical systems with emergent collective computational abilities., Proc. Natl. Acad. Sci. U.S.A. 79 (8) 2554-2558, https://doi.org/10.1073/pnas.79.8.2554 (1982).
   
## **The Model**
In 1982, J. J. Hopfield presented a mathematical framework for how nervous systems could yield content adressable memory. He demonstrated, that computational abilities arise as collective properties of many simple identical units (neurons) and their interconnections. Each neuron can either be firing (1) or resting (0). The general idea is that information corresponds to a state of the system. A state is represented as an array containing the individual states of all neurons V = (V<sub>0</sub>, ... V<sub>N</sub>). For computational purposes, neurons will be converted to bipolar (-1, 1). Each neuron may update itself independently according to a mean attempt rate which results in asynchronous processing. If the weighted sum of all its inputs exceeds a certain threshold, the neuron fires (-> 1), otherwise it will stop firing. To have the system "remember" some memories S<sub>0</sub>, S<sub>1</sub>, ..., a matrix is computed by summing the outer product over all S, thereby encoding how much specific neurons "correlate". This matrix is then used to weigh the inputs in the update rule.

## **The Implementation**
