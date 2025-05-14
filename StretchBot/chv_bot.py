
import cv2, json
import cv2.aruco as aruco
import numpy as np
import glob, datetime, os, sys, math, statistics
import pandas as pd
from pandas import DataFrame
from vision.image import Image
from typing import List, Tuple, Dict
from delta import Delta
from box import Box


class PositionData:
    def __init__(self, names, ids, x, y, z, theta):
        self.names = names
        self.ids = ids
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta

            

def makeDir(name, startPath): # only works for CMD
    now = datetime.datetime.now()
    basePath = os.path.join(startPath, f"{name}_{now.month}.{now.day}_{now.hour}.{now.minute}.{now.second}")
    # basePath = f"C:/Users/irosenstein/Documents/Vision/Projects/AprilTagDetection/PhotoDetection/processing/{name}_{now.month}.{now.day}_{now.hour}.{now.minute}/"
    os.makedirs(basePath,  exist_ok=True)
    filteredPath = os.path.join(basePath, f'processing')
    stepperPath = os.path.join(basePath, f'undistorted')
    tagPath = os.path.join(basePath, f'tagDetection')
    boxPath = os.path.join(basePath, f'boxPosition')
    os.makedirs(filteredPath, exist_ok=True)
    os.makedirs(stepperPath, exist_ok=True)
    os.makedirs(tagPath, exist_ok=True)
    os.makedirs(boxPath, exist_ok=True)
    return basePath, filteredPath, stepperPath, tagPath, boxPath

def initImages(images, setting):
    initImages = []
    for fname in images:
        _, tail = os.path.split(fname)
        name = tail.replace(".JPG", "")
        img = cv2.imread(fname)
        image = Image(img, name, setting)
        initImages.append(image)

    return initImages

def undistortImages(images, savePath):
    undistortedImages = []
    for img in images:

        # img.show() #distorted

        undImg = img.undistort()
        # img.show() #undistorted
        undistortedImages.append(img)
        cv2.imwrite(os.path.join(savePath, f'{img.name}'), undImg)

    return undistortedImages

def saveImage(img, name, setting, savePath):
    cv2.imwrite(os.path.join(savePath, name), img) #correct
    image = Image(img, name, setting)
    return image

def processImages(images, savePath):
    processed = []
    
    for image in images: # not sure if grabbing img or image name
        name = image.name.replace(".jpg", "")
        # img = cv2.imread(fname)

        img = image.image
        setting = image.type
        b, g, r = cv2.split(img)
        h,  w = img.shape[:2]
        ratio = .25
        
        # cv2.namedWindow('Blue', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Blue', int(w*ratio), int(h*ratio))
        # cv2.imshow('Blue', b)

        # cv2.namedWindow('Green', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Green', int(w*ratio), int(h*ratio))
        # cv2.imshow('Green', g)

        # cv2.namedWindow('Red', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Red', int(w*ratio), int(h*ratio))
        # cv2.imshow('Red', r)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayName = f'{name}_a_gray.jpg'
        grayImg = saveImage(gray, grayName, setting, savePath)
        processed.append(grayImg)


        dialated = cv2.dilate(gray, np.ones((7,7), np.uint8))
        dialatedName = f'{name}_b_dialated.jpg'
        diaImg = saveImage(dialated, dialatedName, setting, savePath)
        processed.append(diaImg)

        bg_img = cv2.medianBlur(dialated, 21)
        mbName = f'{name}_c_medianBlur.jpg'
        mbImg = saveImage(bg_img, mbName, setting, savePath)
        processed.append(mbImg)

        diff_img = 255 - cv2.absdiff(gray, bg_img)

        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        normName = f'{name}_d_normalized.jpg'
        normImg = saveImage(norm_img, normName, setting, savePath)
        processed.append(normImg)

        norm_img2 = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        norm2Name = f'{name}_e_norm2.jpg'
        normImg2 = saveImage(norm_img2, norm2Name, setting, savePath)
        processed.append(normImg2)        


        # thresh_img = cv2.adaptiveThreshold(norm_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                    cv2.THRESH_BINARY, 11, 2)
        # show("thresholding", thresh_img)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_img = clahe.apply(norm_img)
        enhName = f'{name}_f_enhanced.jpg'
        enhImg = saveImage(enhanced_img, enhName, setting, savePath)
        processed.append(enhImg)   
    return processed

def detectTags(images, tagSizes, boxTags, botTags, savePath, getPosData=False):
    attempts = []

    nameData = []
    idData = []
    xData = []
    yData = []
    zData = []
    tData = []
    tagSizeBox = tagSizes[0]
    tagSizeBot = tagSizes[1]

    for img in images:



        # success, tags = img.detectTags(tagSize, K, D)
        successTags, tags = img.detectTags()

        for tagID in [tag.id for tag in tags]:
            if tagID in boxTags:
                successPos, tags = img.getWorldPosTags(tagSizeBox)
            elif tagID in botTags:
                successPos, tags = img.getWorldPosTags(tagSizeBot)
            else:
                print(f'tag {tagID} is unregistered')



        cv2.imwrite(os.path.join(savePath, f'{img.name}'), img.image)

        # img.show()
        attempts.append(img)

        if getPosData:

            
            if successPos:
                for tag in tags: 
                    nameData.append(img.name)
                    idData.append(tag.id)
                    xData.append(tag.worldCoord[0])
                    yData.append(tag.worldCoord[1])
                    zData.append(tag.worldCoord[2])
                    tData.append(tag.theta)                
            else: 
                nameData.append(img.name)
                idData.append(None)
                xData.append(None)
                yData.append(None)
                zData.append(None)
                tData.append(None)
    
    positionDataTags = PositionData(nameData, idData, xData, yData, zData, tData)
    return attempts, positionDataTags

def getBoxPosFromTags(tagImg: Image, history, relevantIds=None):
    """this might be bad"""
    tags = tagImg.tags
    tagsPos = {}
    tagIds = [tag.id for tag in tags]

    if relevantIds is not None:
        tags = [tag for tag in tags if tag.id in relevantIds]

    xs = []
    ys = []
    zs = []
    ts = []

    xs, ys, zs = zip(*(tag.worldCoord for tag in tags))
    ts = [tag.theta for tag in tags]
    
    if len(tags) == 4: # all tags recognized

        # for tag in tags: 
        #     x, y, z = tag.worldCoord
        #     t = tag.theta
        #     xs.append(x)
        #     ys.append(y)
        #     zs.append(z)
        #     ts.append(t)        

        avgX = statistics.mean(xs)
        avgY = statistics.mean(ys)
        avgZ = statistics.mean(zs)
        avgT = statistics.mean(ts)

        
        boxPos = [avgX, avgY, avgZ, avgT]
        h = [boxPos, tags]
    
    elif len(set(tagIds) & set([0, 2])) ==2: 
        opposing = [tag for tag in tags if tag.id in [0,2]]
        x1, y1, z1 = opposing[0].worldCoord
        t1 = opposing[0].theta
        x2, y2, z2 = opposing[1].worldCoord
        t2 = opposing[1].theta

        boxPos = [(x1 + x2) / 2 , (y1 + y2) / 2, statistics.mean(zs), statistics.mean(ts)] 
        h = [boxPos, tags]

    elif len(set(tagIds) & set([1, 3])) ==2: # look for opposing sides and find mdpt
        opposing = [tag for tag in tags if tag.id in [1,3]]
        x1, y1, z1 = opposing[0].worldCoord
        t1 = opposing[0].theta
        x2, y2, z2 = opposing[1].worldCoord
        t2 = opposing[1].theta

        boxPos = [(x1 + x2) / 2 , (y1 + y2) / 2, statistics.mean(zs), statistics.mean(ts)] 
        h = [boxPos, tags]
    
    elif len(tags) == 1:  
        diffX = []
        diffY = []
        if not history or history is None:
            print(f'only found one tag and no history')
            return None, None
        single = tags[0]
        # find angle and distance away from center point from history
        # relevantH = [entry for entry in history if single.id in [tag.id for tag in entry[1]]] # check this
        # tagPoses = [tag for tag in history[1] if tag.id == single.id]
        for entry in history: 
            cp = entry[0]
            tags = entry[1]
            t = [tag for tag in tags if tag.id == single.id]
            if len(t) == 1:
                tag = t[0]
                # cpSinglePast.append([cp, (tag.worldCoord[0], tag.worldCoord[1])])
                diffX.append(cp[0] - tag.worldCoord[0])
                diffY.append(cp[1] - tag.worldCoord[1])
                
        
        deltaX = statistics.mean(diffX)
        deltaY = statistics.mean(diffY)
        x1, y1, z1 = single.worldCoord
        t1 = single.theta

        boxPos = [x1 + deltaX, y1 + deltaY, z1, t1]
        h = [boxPos, tags]
    
    else: # no tags dectected
        print(f"no tags detected in {tagImg.name}")
        return None, None

    return boxPos, h
        




def getBoxPosesFromTags(tagImgs: List[Image], undstImgs: List[Image], savePath, relevantIds=None, getPosData=False):
    undstImgs = {img.name: img for img in undstImgs}
    boxPosImages = []

    nameData = []
    idData = []
    xData = []
    yData = []
    zData = []
    tData = []


    for tagImg in tagImgs:
        detectedTags = [tag.id for tag in tagImg.tags]

        if len(detectedTags) > 0: # ensure tags are detected

            undstImg = undstImgs[tagImg.name]
            

            imageBox, [x, y, z, theta] = getBoxPosFromTags(tagImg, undstImg, relevantIds)

            cv2.imwrite(os.path.join(savePath, f'{imageBox.name}'), imageBox.image)
            boxPosImages.append(imageBox)
        else:
            print(f"cannot find box position in {tagImg.name} because no tags were detected")
        
        if getPosData:
            if len(detectedTags) > 0:
                nameData.append(imageBox.name)
                idData.append(detectedTags)
                xData.append(x)
                yData.append(y)
                zData.append(z)
                tData.append(theta)                
            else: 
                nameData.append(imageBox.name)
                idData.append(None)
                xData.append(None)
                yData.append(None)
                zData.append(None)
                tData.append(None)

    posDataBox = PositionData(nameData, idData, xData, yData, zData, tData)

    return boxPosImages, posDataBox
        
def getBoxDeltaFromData(tagImages: List[Image]):
    first = tagImages[0]
    nameData = [first.name]
    idData = [[tag.id for tag in first.tags]]
    xData = [None]
    yData = [None]
    zData = [None]
    tData = [None]

    sorted_images = sorted(tagImages, key=lambda img: int(img.name.split('_')[0]))
    # for img in sorted_images:
    #     print(img.name)

    for i in range(1, len(sorted_images)):
        curr = sorted_images[i]
        prev = sorted_images[i-1]

        # delta
        d = Delta(curr, prev)
        movement = d.calcDeltas()
        

        nameData.append(curr.name)

        idData.append(d.getCurrTagsIDs())

        xData.append(movement[0])
        yData.append(movement[1])
        zData.append(movement[2])
        tData.append(movement[3])

    posDataBox = PositionData(nameData, idData, xData, yData, zData, tData)
    # print(f'POS DATA')
    # print(f'names: {nameData}\nidData: {[[int(id) for id in idList] for idList in idData]}\nxData: {xData}\nyData: {yData}')
    return posDataBox
    
def posDataToDataFrameDelta(posData, output_csv_path=None):
    data_rows = []
    names = posData.names
    idData = posData.ids
    # print(idData)
    xData = posData.x
    yData = posData.y
    zData = posData.z
    thetaData = posData.theta
    for i in range(len(names)):
        # print(idData[i])
        idDataForm = [int(id) for id in idData[i]]
        row = {
            "name": names[i],
            "detected_ids": idDataForm,
            "deltaX": xData[i],
            "deltaY": yData[i],
            "deltaZ": zData[i],
            "deltaTheta": thetaData[i]
        }
        data_rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(data_rows)

    # Export to Excel
    # df.to_excel("position_data.xlsx", index=False)
    return df

def identifyLR(tag, tags=None, history=None):
    left = 0
    right = 1


    if history: 
        mdptX = history[1][0]
        x, y, z = tag.worldCoord
        if x > mdptX: 
            return right
        else:
            return left
    elif tags: 
        x1, y1, z1 = tags[0].worldCoord
        x2, y2, z2 = tags[1].worldCoord

        if x1 < x2: 
            leftTag = tags[0]
            rightTag = tags[1]
        else:
            leftTag = tags[1]
            rightTag = tags[0]
        
        if leftTag.id == tag.id:
            return left
        elif rightTag.id == tag.id:
            return right
        else: 
            return None
           
    else:
        print(f'Not enough info, pls input two tags or history')

def getTowerPos():
    if len(detBotTags) == 2: # enough tags detected
        x1 = detBotTags[0].worldCoord[0]
        y1 = detBotTags[0].worldCoord[1]
        x2 = detBotTags[1].worldCoord[0]
        y2 = detBotTags[1].worldCoord[1]

        dx = x2 - x1
        dy = y2 - y1
        
        towerLength = math.dist([x1, y1], [x2, y2])
        towerMidpoint = ((x1 + x2) / 2 , (y1 + y2) / 2)
        towerAngle = math.degrees(math.atan2(dy,dx))
        towerPos = [towerLength, towerMidpoint, towerAngle]
    elif len(detBotTags) == 1 and history is not None: # use history
        validForHistory = False
        # get current tag to calc position, determine if left or right
        tag = detBotTags[0]
        pastLength = history[0]
        pastMid = history[1]
        pastAngle = history[2]
        if identifyLR(tag, history): # if tag is on the right #TODO check math
            towerMidpoint = (tag.worldCoord[0] - (pastLength/2), (pastLength/2)*math.tan(pastAngle))
        elif identifyLR(tag, history) == 0:
            towerMidpoint = (tag.worldCoord[0] + (pastLength/2), (pastLength/2)*math.tan(pastAngle))
        else: 
            towerMidpoint = None
        towerPos = [pastLength, towerMidpoint, pastAngle] 
    elif not detBotTags or len(detBotTags) ==1 and history is None: 
        print(f'not enough info to get box position relative to bot in {img.name}')
        return None, None, None, None

def boxPosToBot(img: Image, boxTags: List, botTags: List, history):
    """get position of box in relation to two tags on tower
        one img at a time"""
    
    validForHistory = False
    

    # seperate tower tags and box tags
    detBoxTags = [t for t in img.tags if t.id in boxTags]
    detBotTags = [t for t in img.tags if t.id in botTags]

    # get box position
    if detBoxTags: 
        [bx, by, bz, bt], newH = getBoxPosFromTags(img, boxTags)
        history.append(newH)
    else: # if empty, not enough info
        return

    # tower position: length and midpoint
    if len(detBotTags) == 2: # enough tags detected
        x1 = detBotTags[0].worldCoord[0]
        y1 = detBotTags[0].worldCoord[1]
        x2 = detBotTags[1].worldCoord[0]
        y2 = detBotTags[1].worldCoord[1]

        dx = x2 - x1
        dy = y2 - y1
        
        towerLength = math.dist([x1, y1], [x2, y2])
        towerMidpoint = ((x1 + x2) / 2 , (y1 + y2) / 2)
        towerAngle = math.degrees(math.atan2(dy,dx))
        towerPos = [towerLength, towerMidpoint, towerAngle]
    elif len(detBotTags) == 1 and history is not None: # use history
        validForHistory = False
        # get current tag to calc position, determine if left or right
        tag = detBotTags[0]
        pastLength = history[0]
        pastMid = history[1]
        pastAngle = history[2]
        if identifyLR(tag, history): # if tag is on the right #TODO check math
            towerMidpoint = (tag.worldCoord[0] - (pastLength/2), (pastLength/2)*math.tan(pastAngle))
        elif identifyLR(tag, history) == 0:
            towerMidpoint = (tag.worldCoord[0] + (pastLength/2), (pastLength/2)*math.tan(pastAngle))
        else: 
            towerMidpoint = None
        towerPos = [pastLength, towerMidpoint, pastAngle] 
    elif not detBotTags or len(detBotTags) ==1 and history is None: 
        print(f'not enough info to get box position relative to bot in {img.name}')
        return None, None, None, None

            
    # get position of box vs position of tags
    # calculate position of tower
    # calculate position of box

    return [distX, distY, distZ, diffTheta], boxPos, towerPos, validForHistory

def boxPosesToBot(tagImages: List[Image], boxTags: List, botTags: List):
    pastTowerHistory = None 
    tsowerPos = []
    skippedImgs = []

    nameData = []
    idData = []
    xData = []
    yData = []
    zData = []
    tData = []


    for img in tagImages:

        difference, newTowerPos, validForHistory = boxPosToBot(img, boxTags, botTags, pastTowerHistory)
        # towerPosition = [length, midpoint, [left, right]]

        if validForHistory: # both tags were detected and data is good

            # tower history
            tsowerPos.append(newTowerPos)
            nLength = newTowerPos[0]
            nMidpoint = newTowerPos[1]
            angle = newTowerPos[2]

            allMidpoints = [e[1] for e in tsowerPos]
            avgMdptX = statistics.mean([pos[0] for pos in allMidpoints])
            avgMdptY = statistics.mean([pos[1] for pos in allMidpoints])
            allAngles = [a for a in tsowerPos[2]]
            avgAngle = statistics.mean(allAngles)

            pastTowerHistory = [nLength, [avgMdptX, avgMdptY], avgAngle]

            # data collection
            nameData.append(img.name)
            idData.append([tag.id for tag in img.tags])
            xData.append(difference[0])
            yData.append(difference[1])
            zData.append(difference[2])
            tData.append(difference[3])

        elif len(tsowerPos) == 0: # no history on tower yet, wait to add data later
            skippedImgs.append(img)

            nameData.append(img.name)
            idData.append([tag.id for tag in img.tags])

    for img in skippedImgs: # add missing data, CHECK THIS

        difference, newTowerPos = boxPosToBot(img, boxPosData, botTags, pastTowerHistory)

        i = nameData.index(img.name)
        xData.insert(i, difference[0])
        yData.insert(i, difference[1])
        zData.insert(i, difference[2])
        tData.insert(i, difference[3])


    boxToBotPosData = PositionData(nameData, idData, xData, yData, zData, tData)
     
    return boxToBotPosData

if __name__ == '__main__':
    images = glob.glob('C:/Users/irosenstein/Documents/Vision/Photos/CaseHandlingVariation/chv_5_7_25_org_5.8.16.19/cam_2485/*.JPG')
    resultsPath = f'C:/Users/irosenstein/Documents/Vision/Projects/StretchBot/processing'
    name = 'chv_5_7_25__cam_2485'
    botTags = [4,5,6,7]
    boxTags = [0,1,2,3]
    tagSizeBot = 50 # mm
    tagSizeBox = 64.4
    goProSettings = ["photo_linear", "photo_fisheye", "video_fisheye"]
    setting = goProSettings[1]

    basePath, filterPath, undistortPath, tagPath, boxPath = makeDir(name, resultsPath)


    imagesInit = initImages(images, setting)

    undistortedImages = undistortImages(imagesInit, undistortPath)

    # processedImages = processImages(undistortedImages, filterPath)
    # processedImages = glob.glob(f'{filterPath}/*.JPG')

    tagImages, posDataTags = detectTags(undistortedImages, [tagSizeBox, tagSizeBot], boxTags, botTags, tagPath, True) 
    # boxPosImages, posDataBox = getBoxPosesFromTags(tagImages, undistortedImages, boxPath, boxTags)
    boxPosData = boxPosesToBot(tagImages,boxTags, botTags)
    # posDataBox = getBoxDeltaFromData(tagImages)

    df = posDataToDataFrameDelta(posDataBox)
    df.to_excel(f'{basePath}/{name}_DeltaPos.xlsx', sheet_name='sheet1', index=False)

    with pd.ExcelWriter(f'{basePath}/{name}_DeltaPos.xlsx', engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, sheet_name='sheet1', index=False)
        other_df.to_excel(writer, sheet_name='sheet2', index=False)

    cv2.waitKey(0)
    cv2.destroyAllWindows()