# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 23:50 2020

"""

from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np

UPLOAD_FOLDER = './flask app/assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# Create Database if it doesnt exist

app = Flask(__name__,static_url_path='/assets',
            static_folder='./flask app/assets', 
            template_folder='./flask app')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def root():
   return render_template('index.html')

@app.route('/index.html')
def index():
   return render_template('index.html')

@app.route('/prevention.html')
def prevention():
   return render_template('prevention.html')

@app.route('/upload.html')
def upload():
   return render_template('upload.html')

@app.route('/result_chest.html')
def upload_chest():
   return render_template('result_chest.html')

@app.route('/uploaded_chest', methods = ['POST', 'GET'])
def uploaded_chest():
   # if request.method == 'POST':
   #    # check if the post request has the file part
   #    if 'file' not in request.files:
   #       flash('No file part')
   #       return redirect(request.url)
   file = request.files['file']
      # if file.filename == '':
      #    flash('No selected file')
      #    return redirect(request.url)
      # if file:
   file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_chest.jpg'))
   # resnet_chest = load_model('models/resnet_chest.h5')
   # vgg_chest = load_model('models/vgg_chest.h5')
   inception_chest = load_model('models/inception_chest_model.h5')
   # xception_chest = load_model('models/xception_chest.h5')

   image = cv2.imread('./flask app/assets/images/upload_chest.jpg') # read file 
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
   image = cv2.resize(image,(224,224))
   image = np.array(image) / 255
   image = np.expand_dims(image, axis=0)
   
      
   inception_pred = inception_chest.predict(image)
   probability = inception_pred[0]
   print("Inception Predictions:")
   if probability[0] > 0.5:
      inception_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
   else:
      inception_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
   print(inception_chest_pred)
   return render_template('results_chest.html',inception_chest_pred=inception_chest_pred)

if __name__ == '__main__':
   app.secret_key = ".."
   app.run()