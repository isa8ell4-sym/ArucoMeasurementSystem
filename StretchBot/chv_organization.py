""" case handling variation photo organization. organizes photos by gopro id """
import json, datetime, os, sys, glob, random, cv2


def makeDir(name, cams): 
    now = datetime.datetime.now()
    camPaths = {}

    basePath = f'C:/Users/irosenstein/Documents/Vision/Photos/CaseHandlingVariation/{name}_{now.month}.{now.day}.{now.hour}.{now.minute}/'
    try:
        os.makedirs(basePath,  exist_ok=False)
    except:
        basePath = f'C:/Users/irosenstein/Documents/Vision/Photos/CaseHandlingVariation/{name}_{now.month}.{now.day}.{now.hour}.{now.minute}_{random.randint(1, 100)}/'
        os.makedirs(basePath,  exist_ok=False)

    for i, camID in enumerate(cams):
        camPaths[f'{camID}'] = os.path.join(basePath, f'cam_{camID}')
        os.makedirs(camPaths[f'{camID}'],  exist_ok=False)

    return basePath, camPaths

def orgImagesIds(images, ids, camPaths):

    for fname in images:
        _, tail = os.path.split(fname)
        name = tail.replace(".jpg", "")
        lastFour = name[-4:]
        img = cv2.imread(fname)
        # print(f'img {name}, {lastFour}')

        for id in ids:
            # print(f'checking ID {id}')
            if id in name:
                # print(f'path: {camPaths[lastFour]}')
                path = os.path.join(camPaths[lastFour], tail)
                print(f'path: {path}/n')

                cv2.imwrite(path, img)





name = 'chv_5_7_25_org'
cameraIDs = ['2485', '0168', '5499']
images = glob.glob('C:/Users/irosenstein/Documents/Vision/Photos/CaseHandlingVariation/chv_5_7_25/*.JPG')

basePath, camPaths = makeDir(name, cameraIDs)
orgImagesIds(images, cameraIDs, camPaths)

