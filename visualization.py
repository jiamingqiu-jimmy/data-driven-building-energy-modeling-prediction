import numpy as np
import pickle
from PIL import Image
import os


def tempToColor(temp):
    maxTemp = 100
    minTemp = 50
    midTemp = (maxTemp - minTemp) / 2
    if temp >= maxTemp:
        return np.array([255, 0, 0])
    elif temp >= midTemp:
        grad = (temp - midTemp) / (maxTemp - midTemp)
        return np.array([255 * grad, 255 * (1 - grad), 0])
    elif temp >= minTemp:
        grad = (temp - minTemp) / (midTemp - minTemp)
        return np.array([0, 255 * grad, 255 * (1 - grad)])
    return np.array([0, 0, 255])


def visualizeData(data, fileout):
    im = Image.open('D:\\Documents\\School\\ECE 228\\Project\\images\\imageDataRoomsOnlyNumbered.png')
    # fileName = open()
    roomLocations = pickle.load('images\\roomLocations.pkl', 'rb')

    frames = []
    newIm = im
    frames.append(newIm)
    for time in data:
        for room in time:
            for i in range(roomLocations[room][0], roomLocations[room][2]):
                for j in range(roomLocations[room][1], roomLocations[room][3]):
                    newIm[i][j] = tempToColor(time[room])
        frames.append(newIm)



    frames[0].save(fileout, save_all=True, append_images=frames[1:], optimize=True, duration=300, loop=0)
