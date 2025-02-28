import mplcyberpunk
import matplotlib.pyplot as plt
import math
import numpy as np
from numpy import transpose

plt.style.use('cyberpunk')
plt.figure(figsize = (10,10))

# For a beamsplitter with fixed angle of incidence
def simpleBeamSplitter():
    inputState = np.matrix([[0],[1]])
    #matrix45 = MC* np.matrix([[1, 1j],[1j, 1]])
    matrix = np.array([[math.cos(0), math.sin(0) * 1j], [math.sin(0) * 1j, math.cos(0)]])
    outputState = np.dot(matrix,inputState)
    return outputState

print(simpleBeamSplitter())

def angleDependentBeamSplitter(inputState = np.array([[0],[1]]), angleTheta = 0):
    matrix = np.array([[math.cos(angleTheta), math.sin(angleTheta) * 1j], [math.sin(angleTheta) * 1j, math.cos(angleTheta)]])
    outputState = np.dot(matrix, inputState)
    return outputState

#print(angleDependentBeamSplitter(anglePhi = 90))

def fidelity(inputState = np.array([[0],[1]]), angleTheta = 0):
    outputState = angleDependentBeamSplitter(inputState, angleTheta)
    target = simpleBeamSplitter()
    return np.matrix(abs(np.dot(transpose(target), outputState)) ** 2).item(0)


set = []
for x in range(360):
    set.append(fidelity(angleTheta= math.radians(x)))

plt.plot(set)
plt.xlabel('Angle Degrees')
plt.ylabel('Normalized fidelity of beam splitter output 1')
plt.title('Fidelity of beam splitter outputs ')
plt.show()
