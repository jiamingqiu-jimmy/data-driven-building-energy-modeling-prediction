import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageTk
import pickle





def findRooms(image, roomNumbers=[]):
    imNumpy = np.asarray(ImageOps.grayscale(image))
    imDims = np.shape(imNumpy)

    black = 0
    white = 255
    def isTopCorner(i,j):
        if imNumpy[i][j] == white and imNumpy[i-1][j] == black and imNumpy[i][j-1] == black:
            return True
        else:
            return False

    def findBotCorner(i,j):
        width = 0
        height = 0
        while imNumpy[i][j+width+1] == white:
            width += 1
            if imNumpy[i-1][j+width] == white or j+width > imDims[1]:
                return None

        while imNumpy[i+height][j+width] == white:
            height += 1
            if imNumpy[i+height-1][j+width+1] == white or i + height > imDims[0]:
                return None
        return i+height, j+width+1

    roomCorners = dict()
    idx = 0
    for i in range(1,imDims[0]):
        for j in range(1,imDims[1]):
            # if idx >= len(roomNumbers):
            #     break
            if isTopCorner(i,j):
                end = findBotCorner(i,j)
                if end != None:
                    roomCorners[str(roomNumbers[idx])] = (i-1,j-1,*end)
                    idx += 1

    return roomCorners

def testColorRooms(image,rooms):

    image = image.convert("RGB")
    imNumpy = np.copy(np.array(image))
    #print(len(rooms.values()))
    col = 0
    for key in rooms:
        for i in range(rooms[key][0], rooms[key][2]):
            for j in range(rooms[key][1], rooms[key][3]):
                if False in (imNumpy[i][j] == np.array([0,0,0])):
                    imNumpy[i][j] = [(col*10)%205+50, (col*75)%205+50, col%205+50]
        col += 1

        # imS = cv2.resize(imNumpy, (860,600))
        # print(key)
        # cv2.imshow("output", imS)
        # cv2.waitKey(0)
    imPil = Image.fromarray(imNumpy)
    return imPil

def simplifyImage(image, thres):
    imNumpy = np.array(ImageOps.grayscale(image))
    dims = np.shape(image)
    for i in range(dims[0]):
        for j in range(dims[1]):
            if imNumpy[i][j] <= thres:
                imNumpy[i][j] = 0
            else:
                imNumpy[i][j] = 255
    imPil = Image.fromarray((imNumpy))
    return imPil

def white_to_transparency(img):
    x = np.asarray(img.convert('RGBA')).copy()

    x[:, :, 3] = (255 * (x[:, :, :3] != 255).any(axis=2)).astype(np.uint8)

    return Image.fromarray(x)

def testGif(imageFull,rooms):
    imColored = testColorRooms(imageFull, rooms)
    frames = [imColored]

    def coolImage(newIm):
        newIm = np.array(newIm)
        finishBool = True
        for key in rooms:
            for i in range(rooms[key][0], rooms[key][2]):
                for j in range(rooms[key][1], rooms[key][3]):
                    if False in (newIm[i][j] == np.array([0,0,0])):
                        if np.sum(newIm[i][j]) < 255 * 3:
                            finishBool = False
                            for k in range(3):
                                if newIm[i][j][k] * 1.3 <= 255:
                                    newIm[i][j][k] = newIm[i][j][k] * 1.3 + 1
                                else:
                                    newIm[i][j][k] = 255

        newIm = Image.fromarray(newIm)
        return newIm, finishBool

    image = imColored
    while True:
        image, done = coolImage(image)
        frames.append(image)
        if done:
            break
        if len(frames) >= 4:
            break

    frames[0].save('images\\testGif.gif', save_all=True, append_images=frames[1:], optimize=True, duration=300, loop = 0)




if __name__ == '__main__':

    # imFile = 'D:\\Documents\\School\\ECE 228\\Project\\images\\imageDataRoomsOnlyNumbered.png'
    # im = Image.open(imFile)
    # im = simplifyImage(im, 254)
    # im.show()
    # im.save('D:\\Documents\\School\\ECE 228\\Project\\images\\imageDataRoomsOnlyNumbered.png')

    im = Image.open('D:\\Documents\\School\\ECE 228\\Project\\images\\imageDataRoomsOnly.png')

    roomNumbers = ['403', '405', '409', '413', '415', '402', '417', '484', '470', '420', '469', '423', '450', '462', '463', '461', '448', '445', '432', '428', '429', '436',
                   '304', '308', '314', '208', '213', '216', '290', '302', '375', '281', '317', '387', '217', '386', '288', '280', '380', '371', '271', '368', '268', '264',
                   '221', '223', '323', '350', '340', '249', '252', '254', '261', '362', '363', '361', '227', '348', '345', '248', '247', '245', '240', '228', '104', '105',
                   '229', '328', '329', '107', '330', '336', '108', '110', '114', '122', '121', '120']

    # roomLocations = findRooms(im, roomNumbers)
    # with open('images\\roomLocations.pkl', 'wb') as f:
    #     pickle.dump(roomLocations, f)
    fileName = open('images\\roomLocations.pkl', 'rb')
    roomLocations = pickle.load(fileName)

    #help 353,437, 453, 490

    # imNumpy = np.array(im.convert("RGB"))
    # imNumpy[1012, 387] = [255,0,0]
    # imNumpy[1044, 428] = [255,0,0]
    # imNumpy[961, 496] = [0,255,0]
    # imNumpy[1094, 552] = [0,255,0]
    # imNumpy[961, 553] = [0,0,255]
    # imNumpy[1030, 636] = [0,0,255]
    # cv2.imshow("output", imNumpy)
    # cv2.waitKey(0)

    # temp = roomLocations['268']
    # roomLocations.pop('268')
    # roomLocations['264'] = (temp[0], temp[1], roomLocations['264'][2], roomLocations['264'][3])
    # roomLocations['268'] = temp
    #
    # with open('images\\roomLocations.pkl', 'wb') as f:
    #     pickle.dump(roomLocations, f)
    fileName = open('images\\roomLocations.pkl', 'rb')
    roomLocations = pickle.load(fileName)

    im = Image.open('D:\\Documents\\School\\ECE 228\\Project\\images\\imageDataRoomsOnlyNumbered.png')
    testGif(im,roomLocations)


    # numberIm = white_to_transparency(Image.open('D:\\Documents\\School\\ECE 228\\Project\\images\\imageDataRoomsOnlyNumbered.png'))
    # colorIm = testColorRooms(im, roomLocations)
    # colorIm.paste(numberIm, (0,0), numberIm)
    # finalIm = colorIm
    # finalIm.show()
    a=5