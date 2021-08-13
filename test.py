"""
|Dounia Hammou(1), Sid Ahmed Fezza(1) and Wassim Hamidouche(2)
|(1) LION Team, National Institute of Telecommunications and ICT, Oran, Algeria
|{dhammou,sfezza}@inttic.dz
|(2) Univ. Rennes, INSA Rennes, CNRS, IETR - UMR 6164, Rennes, France
|wassim.hamidouche@insa-rennes.fr
"""

def process_img(path):
  # This function is used to process each image so that the output would be identical to the vgg16 input
  img = load_img(path,target_size=(224,224)) # each image is firstly loaded with the shape (224,224,3)
  img = img_to_array(img)
  img = np.expand_dims(img,axis=0) # Then each image will expand to this shape (1,224,224,3)
  img = preprocess_input(img) # Then it will be processed to vary between -1 and 1, instead of 0 and 255
  return img

def extract_features(ref_img,dis_img,model):
  # This function is used to extract the necessary features from the reference and distorted images
  # Firstly we extract the features of the reference and distorted images using vgg16
  features1 = model.predict(ref_img)
  features2 = model.predict(dis_img)
  # then we return the absolute value of the difference between the features of the two images
  return np.abs(features1-features2)

def load_models():
  pretrained_model = load_model('models/Pretrained_model.h5',compile=False) # Load the pretrained model

  # Load the boost regressors
  model1 = pickle.load(open('models/xgboost_model.sav', 'rb')) 
  model2 = pickle.load(open('models/lightgbm_model.sav', 'rb'))
  model3 = pickle.load(open('models/catboost_model.sav', 'rb'))

  models = [model1,model2,model3]

  return pretrained_model,models

if __name__ == "__main__": 

  import argparse
  import sys
  import os

  # Define the code arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("ref_path", help="The path to the reference images")
  parser.add_argument("dis_path", help="The path to the distorted images")
  args = parser.parse_args()

  ref_path = args.ref_path
  dis_path = args.dis_path

  if not(os.path.isdir(ref_path)) or not(os.path.isdir(dis_path)) :
    print('The path does not exist, please enter a valid path to the images')
    sys.exit()

  
  # Import the necessary packages
  import numpy as np
  import time
  import pickle
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
  import tensorflow as tf
  from tensorflow.keras.models import load_model 
  from tensorflow.keras.applications.vgg16 import preprocess_input
  from tensorflow.keras.preprocessing.image import load_img , img_to_array

  
  # Load the necessary models 
  pretrained_model,models = load_models()

  start = time.time() # Calculate the time of the start

  f= open("output.txt","w") # Create an empty file with the name output.txt

  number_images = len(os.listdir(dis_path)) # Calculate the number of distorted images

  for i,dis_img in enumerate(sorted(os.listdir(dis_path))):  # Using the distorted path and processing all distorted images
    ref_img = dis_img.split('_')[0]+'.bmp' # Find the name of reference image from the distorted image one
    
    ref_image = process_img(os.path.join(ref_path,ref_img)) # Process the reference image
    dis_image = process_img(os.path.join(dis_path,dis_img)) # Process the distorted image

    features = extract_features(ref_image,dis_image,pretrained_model) # Extract the features using the pretrained model from both the reference and distorted images

    predictions=[]
    for model in models:
      predictions.append(model.predict(features)[0]) # Using the three boost regressors to estimate the image quality

    predictions = np.mean(np.array(predictions),axis=0) # The final image quality score is the mean of predictions of each model (regressor)

    # save the prediction on the text file
    if i==(number_images-1): f.write(dis_img+','+str(predictions))
    else: f.write(dis_img+','+str(predictions)+'\n')

  f.close() # Close the text file

  end = time.time() # Calculate the time of the end

  print('The required time to estimate the quality of one image is: '+str((end-start)/number_images)+' seconds') # display runtime at test per image