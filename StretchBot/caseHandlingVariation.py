
import cv2, json
import cv2.aruco as aruco
import numpy as np
import glob, datetime, os, sys
from pandas import DataFrame
import re
from vision.image import Image


class PositionData:
    def __init__(self, names, ids, x, y, z, theta):
        self.names = names
        self.ids = ids
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta


def makeDir(name): # only works for CMD
    now = datetime.datetime.now()

    basePath = f"C:/Users/irosenstein/Documents/Vision/Projects/AprilTagDetection/PhotoDetection/processing/{name}_{now.month}.{now.day}_{now.hour}.{now.minute}/"
    os.makedirs(basePath,  exist_ok=True)
    filteredPath = os.path.join(basePath, f'processing')
    stepperPath = os.path.join(basePath, f'undistorted')
    detectionPath = os.path.join(basePath, f'tagDetection')
    os.makedirs(filteredPath, exist_ok=True)
    os.makedirs(stepperPath, exist_ok=True)
    os.makedirs(detectionPath, exist_ok=True)
    return basePath, filteredPath, stepperPath, detectionPath

def processImagesOld(images, filteredPath, stepperPath, env='CMD'):
    # process images for better detection
    for fname in images:
        # print(fname)
        _, tail = os.path.split(fname)
        name = tail.replace(".JPG", "")
        # print(name)

        # Load the image
        # image = cv2.imread(fname)
        # undImg = undistortPaddingWidePhotos(fname, 1.5)
        undImg = undistortFisheyeCrop(fname)
        # undImg = undistortImagesLinear(fname)

        os.chdir(filteredPath)
        # print(os.listdir(filteredPath))  
        cv2.imwrite(f'{name}_undistorted.JPG', undImg)

        gray = cv2.cvtColor(undImg, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'{name}_gray.JPG', gray)

        dialated = cv2.dilate(gray, np.ones((7,7), np.uint8))
        cv2.imwrite(f'{name}_dialted.JPG', dialated)
        # show("dialated", dialated)

        bg_img = cv2.medianBlur(dialated, 21)
        # show("bg", bg_img)
        cv2.imwrite( f'{name}_blur.JPG', bg_img)

        diff_img = 255 - cv2.absdiff(gray, bg_img)

        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # show("normalized", norm_img)
        cv2.imwrite(f'{name}_n1.JPG', norm_img)

        norm_img2 = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # show("normalizedGray", norm_img2)
        cv2.imwrite(f'{name}_n2.JPG', norm_img2)

        # thresh_img = cv2.adaptiveThreshold(norm_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                    cv2.THRESH_BINARY, 11, 2)
        # show("thresholding", thresh_img)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_img = clahe.apply(norm_img)
        cv2.imwrite(f'{name}_clache.JPG', enhanced_img)

        # os.chdir(stepperPath)
        cv2.imwrite(os.path.join(stepperPath, f'{name}.JPG'), undImg)
        # i+=1
        # show("clache", enhanced_img)
    #TODO should make return images

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

def detectTags(images, tagSize, savePath, getPosData=False):
    attempts = []

    nameData = []
    idData = []
    xData = []
    yData = []
    zData = []
    tData = []

    for img in images:

        K, D, DIM, R = img.getIntrinsicParams()

        success, tags = img.detectTags(tagSize, K, D)
        cv2.imwrite(os.path.join(savePath, f'{img.name}'), img.image)

        # img.show()
        attempts.append(img)

        if getPosData:
            nameData.append(img.name)
            if success:
                for tag in tags: 
                    idData.append(tag.id)
                    xData.append(tag.worldCoord[0])
                    yData.append(tag.worldCoord[1])
                    zData.append(tag.worldCoord[2])
                    tData.append(tag.theta)                
            else: 
                idData.append(None)
                xData.append(None)
                yData.append(None)
                zData.append(None)
                tData.append(None)
    
    positionData = PositionData(nameData, idData, xData, yData, zData, tData)
    return attempts, positionData

def organize_position_data(positionData, output_csv_path=None):
    # Create a DataFrame from PositionData
    df = DataFrame({
        "Img Name": positionData.names,
        "Tag ID": positionData.ids,
        "x": positionData.x,
        "y": positionData.y,
        "z": positionData.z,
        "theta": positionData.theta
    })

    # Extract prefix number and suffix (last 4 digits)
    df["prefix"] = df["Img Name"].apply(lambda x: int(x.split("_")[0]))
    df["suffix"] = df["Img Name"].apply(
        lambda x: re.search(r'_(\d{4})', x).group(1) if re.search(r'_(\d{4})', x) else None
    )

    # Sort by suffix and then prefix
    df_sorted = df.sort_values(by=["suffix", "prefix"]).reset_index(drop=True)

    # Optionally save to CSV
    if output_csv_path:
        df_sorted.to_csv(output_csv_path, index=False)

    return df_sorted

if __name__ == '__main__':
    images = glob.glob('C:/Users/irosenstein/Documents/Vision/Photos/CaseHandlingVariation/chv_4_15_25_5.5.9.6/cam_2485/*.JPG')
    name = 'chv_4_15_25__cam2485'
    tagSize = 100 # mm
    goProSettings = ["photo_linear", "photo_fisheye", "video_fisheye"]
    setting = goProSettings[0]

    basePath, filterPath, stepperPath, detectionPath = makeDir(name)


    imagesInit = initImages(images, setting)

    undistortedImages = undistortImages(imagesInit, stepperPath)


    processedImages = processImages(undistortedImages, filterPath)

    # processedImages = glob.glob(f'{filterPath}/*.JPG')

    attemptsImages, posData = detectTags(undistortedImages, tagSize, detectionPath, True) ######################## where you left off

    posDataOrg = organize_position_data(posData)
    posDataOrg.to_excel(f'{basePath}/{name}_PositionData.xlsx', sheet_name='sheet1', index=False)

    cv2.waitKey(0)
    cv2.destroyAllWindows()