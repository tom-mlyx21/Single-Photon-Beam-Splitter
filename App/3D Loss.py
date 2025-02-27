import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from numpy import transpose
import mplcyberpunk

import plotly.graph_objects as go

import pandas as pd

plt.style.use('cyberpunk')
plt.figure(figsize = (10,10))

def realSurfaceTarget():
    pass


def realFidelitySurface(angleTheta, angleGamma):
    pass

target = realSurfaceTarget()
set = []
for x in range(360):
    for y in range(360):
        set.append(realFidelitySurface(math.radians(x), math.radians(y)))


plt.plot(set)
plt.xlabel('Angle Degrees')
plt.ylabel('Normalized fidelity of beam splitter output 1')
plt.title('Fidelity of beam splitter outputs ')
plt.show()