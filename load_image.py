import cv2, os, sys
import numpy as np

os.chdir(sys.path[0])

def load_image(dataPath):
    dataset = []
    answer = {}
    explain = {
        0:'Speed limit (20 km/h)',
        1:'Speed limit (30 km/h)', 
        2:'Speed limit (50 km/h)', 
        3:'Speed limit (60 km/h)', 
        4:'Speed limit (70 km/h)', 
        5:'Speed limit (80 km/h)', 
        6:'End of speed limit (80 km/h)', 
        7:'Speed limit (100km/h)', 
        8:'Speed limit (120km/h)', 
        9:'No passing', 
        10:'No passing veh over 3.5 tons', 
        11:'Right-of-way at intersection', 
        12:'Priority road', 
        13:'Yield', 
        14:'Stop', 
        15:'No vehicles', 
        16:'Veh > 3.5 tons prohibited', 
        17:'No entry', 
        18:'General caution', 
        19:'Dangerous curve left', 
        20:'Dangerous curve right', 
        21:'Double curve', 
        22:'Bumpy road', 
        23:'Slippery road', 
        24:'Road narrows on the right', 
        25:'Road work', 
        26:'Traffic signals', 
        27:'Pedestrians', 
        28:'Children crossing', 
        29:'Bicycles crossing', 
        30:'Beware of ice/snow',
        31:'Wild animals crossing', 
        32:'End speed + passing limits', 
        33:'Turn right ahead', 
        34:'Turn left ahead', 
        35:'Ahead only', 
        36:'Go straight or right', 
        37:'Go straight or left', 
        38:'Keep right', 
        39:'Keep left', 
        40:'Roundabout mandatory', 
        41:'End of no passing', 
        42:'End no passing veh > 3.5 tons',
        43:'Speed limit (5 km/h)',
        46:'Speed limit (40 km/h)',
        52:'',
        53:'',
        54:'',
        55:'',
        56:'',
        58:'',
        59:'',
        69:'',
        71:'',
        76:'',
        84:'',
        95:'',
        96:'',
        97:'',
    }
    folder = os.listdir(dataPath)
    ID = 0
    for folder_name in folder:
        files = os.listdir(dataPath + '/' + folder_name)
        count = 0
        answer[ID] = explain[folder_name]
        for name in files:
            img = cv2.imread(dataPath + '/' + folder_name + '/' + name)
            img = cv2.resize(img, (36, 16))
            dataset.append((np.array(img), ID))
            count = count + 1
            if count >= 1000:
                break
        ID = ID + 1

    return dataset, answer
"""
if __name__ == '__main__':
    dataPath = 'C:/Users/User/Desktop/final_project/used/Train'
    load_image(dataPath)
    print('done')
"""
