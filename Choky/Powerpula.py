import torch
import cv2
import numpy as np
from PIL import Image,ImageDraw
import sys, os
sys.path.append(r"C:\ProJ\yolov5")

model = torch.hub.load('.', 'custom', path="best.pt", source='local').to('cuda')
model.conf = 0.50
model.eval()

def Prediction(Image_all):
    X = model(Image_all)
    return X


def Croping_Counter_Codination(X,Image_all):
    colony_count = []
    colony_codinate = {}

    for I in range(len(Image_all)):
        predictions = X.pred[I]
        img = np.array(Image_all[I])
        colony_count.append(len(predictions))
        codination = []

        for J in range(len(predictions)):
            xmin, ymin, xmax, ymax = predictions[J][:4].tolist()

            Image_all[I] = cv2.rectangle(img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(255, 105, 250),1) # rectangle

            text = str(J+1)
            font = cv2.FONT_HERSHEY_PLAIN 
            font_scale = 1
            font_thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            center_x = (int(xmin) + int(xmax)) // 2
            center_y = (int(ymin) + int(ymax)) // 2
            text_origin = (center_x - text_size[0] // 2, center_y + text_size[1] // 2)
            cv2.putText(img, text, text_origin, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            

            new_width = 700/Image_all[I].shape[1]
            new_height = 650/Image_all[I].shape[0]

            codination.append([int(xmin*new_width),int(ymin*new_height), int(xmax*new_width)-int(xmin*new_width),int(ymax*new_height)-int(ymin*new_height)])
            #codination.append([int(xmin),int(ymin), int(xmax),int(ymax)])
            
        colony_codinate[I] = codination
        
    Image_all = [Image.fromarray(I) for I in Image_all]

    return Image_all,colony_count,colony_codinate