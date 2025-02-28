import numpy as np
import math
from math import e
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from numpy import transpose
import mplcyberpunk

import matplotlib.animation as anim
from matplotlib import cm
from matplotlib.animation import FFMpegWriter, PillowWriter

plt.style.use('cyberpunk')
plt.figure(figsize = (10,10))


# All functions duplicated due to different requirements pending an imaginary state
def simpleSurfaceTarget():
    U = np.array([[math.cos(0), math.sin(0) * 1j], [math.sin(0) * 1j, math.cos(0)]])
    inputState = np.array([[math.cos(0)], [e**(0+1j)*math.sin(0)]])
    Target = np.dot(U, inputState)
    return Target

def complexSurfaceTarget():
    U = np.array([[math.cos(0), math.sin(0) * 1j], [math.sin(0) * 1j, math.cos(0)]])
    inputState = np.array([[math.cos(0)], [e**(0+2*math.pi*1j)*math.sin(0)]])
    Target = np.dot(U, inputState)
    return Target

def simpleSurface(gamma, theta):
    U = np.array([[math.cos(gamma), math.sin(gamma) * 1j], [math.sin(gamma) * 1j, math.cos(gamma)]])
    inputState = np.array([[math.cos(theta/2)], [e**(0+1j)*math.sin(theta/2)]])
    outPut = np.dot(U, inputState)
    return outPut

def complexSurface(gamma, theta, phi):
    U = np.array([[math.cos(gamma), math.sin(gamma) * 1j], [math.sin(gamma) * 1j, math.cos(gamma)]])
    inputState = np.array([[math.cos(theta/2)], [e**(0+phi*1j)*math.sin(theta/2)]])
    outPut = np.dot(U, inputState)
    return outPut

def simpleSurfaceFidelity(gamma, theta):
    outputState = simpleSurface(gamma, theta)
    target = simpleSurfaceTarget()
    return np.matrix(abs(np.dot(transpose(target), outputState)) ** 2).item(0)

def complexSurfaceFidelity(gamma, theta, phi):
    outputState = complexSurface(gamma, theta, phi)
    target = complexSurfaceTarget()
    return np.matrix(abs(np.dot(transpose(target), outputState)) ** 2).item(0)

def simpleArrayCruncher(X, Y):
    out = []

    # Theta range 180, as the half spherical coordinate
    for x in range(180):
        mid = []
        # Gamma range 360, as the full 2pi of the matrix
        for y in range(360):
            gamma = X[x][y]
            theta = Y[x][y]
            mid.append(simpleSurfaceFidelity(math.radians(gamma), math.radians(theta)))
        out.append(mid)
    return np.array(out)

def complexArrayCruncher(X, Y, phi):
    out = []
    # Theta range 180, as the half spherical coordinate
    for x in range(180):
        mid = []
        # Gamma range 360, as the full 2pi of the matrix
        for y in range(360):
            gamma = X[x][y]
            theta = Y[x][y]
            mid.append(complexSurfaceFidelity(math.radians(gamma), math.radians(theta), math.radians(phi)))
        out.append(mid)
    return np.array(out)

def simpleSurfacePlot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.arange(0, 360, 1)
    Y = np.arange(0, 180, 1)
    X, Y = np.meshgrid(X, Y)
    Z = simpleArrayCruncher(X, Y)

    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('Gamma')
    ax.set_ylabel('Theta')
    ax.set_zlabel('Fidelity')

    plt.show()

def complexSurfacePlot(phi):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.arange(0, 360, 1)
    Y = np.arange(0, 180, 1)
    X, Y = np.meshgrid(X, Y, )
    Z = complexArrayCruncher(X, Y, phi)

    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('Gamma')
    ax.set_ylabel('Theta')
    ax.set_zlabel('Fidelity')

    plt.show()

def simpleHelper():
    X = np.arange(0, 360, 1)
    Y = np.arange(0, 180, 1)
    X, Y = np.meshgrid(X, Y)
    Z = simpleArrayCruncher(X, Y)
    return X, Y, Z


def complexHelper(phi):
    X = np.arange(0, 360, 1)
    Y = np.arange(0, 180, 1)
    X, Y = np.meshgrid(X, Y)
    Z = complexArrayCruncher(X, Y, phi)
    return X, Y, Z

def animatedFidelity():
    plt.rcParams['animation.ffmpeg_path'] = 'C:/ffmpeg-2025/bin/ffmpeg.exe'

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    metadata = dict(title='Single Photon Beam Splitter Fidelity in 4 Dimensions', artist='TomM')
    #writer = PillowWriter(fps=15, metadata=metadata)
    writer = FFMpegWriter(fps=15, metadata=metadata)

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.set_title('Single Photon Beam Splitter Fidelity in 4 Dimensions')
    ax.set_xlabel('Gamma')
    ax.set_ylabel('Theta')
    ax.set_zlabel('Fidelity')
    # No label for phi, as this evolves through time

    with writer.saving(fig, "exp3d.mp4", 100):
        for xval in np.linspace(0, 360, 360):
            print(xval)
            xlist, ylist, zlist = complexHelper(xval)
            ax.set_zlim(0, 1)
            ax.plot_surface(xlist, ylist, zlist, cmap=cm.viridis)
            ax.set_title('Single Photon Beam Splitter Fidelity in 4 Dimensions')
            ax.set_xlabel('Angle of Input (Degrees)')
            ax.set_ylabel('Theta angle (Degrees)')
            ax.set_zlabel('Fidelity')
            writer.grab_frame()
            plt.cla()






#simpleSurfacePlot()
animatedFidelity()