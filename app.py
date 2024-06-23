import torch
import cv2
import numpy as np
from PIL import Image,ImageDraw
from flask import Flask,request,jsonify,send_file
from flask_cors import CORS
from io import BytesIO


import sys, os
sys.path.append(r"C:\ProJ\yolov5\Choky")

from Choky.Powerpula import Prediction,Croping_Counter_Codination

app = Flask(__name__)
CORS(app)

@app.route("/")
async def index():
    return {"message":"ok"},200

@app.route("/UpFilesImages",methods=['POST']) # ผู้ใช้อัพโหลดไฟล์  [Images]
async def UpFilesImages():  
    if request.method == 'POST':
        Image_all = []
        files = request.files.getlist('Files_image')   
        for file in files:
             image = Image.open(BytesIO(file.read()))
             Image_all.append(image)

        X = Prediction(Image_all)
        X,Y,Z = Croping_Counter_Codination(X,Image_all)

    # for I in range(len(files)):
    #      X[I].save(f"{files[I].filename}")

    return {"message":"ok","counter": Y ,"codination": Z },200

if __name__ == '__main__':
    app.run(debug=True)
