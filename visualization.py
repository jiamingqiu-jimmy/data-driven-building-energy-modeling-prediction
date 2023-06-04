import numpy as np
import pickle
from PIL import Image
import os

class tempToColor():

    def __init__(self, max_temp, min_temp):
        self.max_temp = max_temp-5
        self.min_temp = min_temp
        self.mid_temp = (max_temp + min_temp)/2

    def findColor(self, temp):
        if temp >= self.max_temp:
            return np.array([255, 0, 0])
        elif temp >= self.mid_temp:
            grad = (temp - self.mid_temp) / (self.max_temp - self.mid_temp)
            return np.array([255 * grad, 255 * (1 - grad), 0])
        elif temp >= self.min_temp:
            grad = (temp - self.min_temp) / (self.mid_temp - self.min_temp)
            return np.array([0, 255 * grad, 255 * (1 - grad)])
        return np.array([0, 0, 255])


def visualizeData(data, fileout, visual_room_order):
    im = Image.open('D:\\Documents\\School\\ECE 228\\Project\\images\\imageDataRoomsOnlyNumbered.png')
    fileName = open('images\\roomLocations.pkl', 'rb')
    roomLocations = pickle.load(fileName)


    tempColor = tempToColor(np.max(data), np.min(data))
    print(np.max(data), np.min(data))

    frames = []
    newIm = np.array(im.convert('RGB'))
    frames.append(Image.fromarray(newIm))
    for time in range(data.shape[0]):
        if time % 1 == 0:
            for idx, room in enumerate(visual_room_order):
                for i in range(roomLocations[room][0], roomLocations[room][2]):
                    for j in range(roomLocations[room][1], roomLocations[room][3]):
                        if False in (newIm[i][j] == np.array([0, 0, 0])):
                            newIm[i][j] = tempColor.findColor(data[time][idx])
            frames.append(Image.fromarray(newIm))
        if time % 20 == 0:
            print('time compeleted: {}'.format(time))



    frames[1].save(fileout, save_all=True, append_images=frames[2:], optimize=True, duration=8, loop=0)


def reorder_data(data, data_room_order, visual_room_order):

    dataNew = np.zeros((data.shape[0], len(visual_room_order)))
    for room_idx in range(len(visual_room_order)):
        data_idx = np.where(data_room_order == visual_room_order[room_idx])[0][0]
        dataNew[:,room_idx] = data[:,data_idx]
    return dataNew

data = np.load('preprocessing_output\\merged_temps_time_30T.npy')[0:500,:]
# data[783][37] = 73.61798
data_room_order = np.load('preprocessing_output\\merged_rooms_list.npy')

visual_room_order = ['403', '405', '409', '413', '415', '402', '417', '484', '470', '420', '469', '423', '450', '462', '463', '461', '448', '445', '432', '428', '429', '436',
                     '304', '308', '314', '208', '213', '216', '290', '302', '375', '281', '317', '387', '217', '386', '288', '280', '380', '371', '368', '268', '264', '221',
                     '223', '323', '350', '340', '249', '252', '254', '261', '362', '363', '361', '227', '348', '345', '248', '247', '245', '240', '104', '105', '229', '328',
                     '329', '107', '330', '336', '108', '110', '114', '122', '121', '120']

data_ordered = reorder_data(data, data_room_order, visual_room_order)
visualizeData(data_ordered, 'images\\previewTemps.gif', visual_room_order)