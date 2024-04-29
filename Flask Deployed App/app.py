import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
from transformers import AutoImageProcessor, AutoModelForImageClassification

app = Flask(__name__)

disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')
model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()
processor = AutoImageProcessor.from_pretrained("linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")
model = AutoModelForImageClassification.from_pretrained("linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")

# Mapping of class indices to disease names
class_names = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    3: 'Apple___healthy',
    2: 'Apple___Cedar_apple_rust',
    4: 'Blueberry___healthy',
    5: 'Cherry___Powdery_mildew''Background_without_leaves',
    6: 'Cherry___healthy',
    7: 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn___Common_rust',
    9: 'Corn___Northern_Leaf_Blight',
    10: 'Corn___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    17: 'Peach___healthy',
    16: 'Peach___Bacterial_spot',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy',
    38: 'Background_without_leaves'
}

def prediction(image_path):
        # Get the uploaded image file
        if image_path != '':
            # Load and preprocess the image
            img = Image.open(image_path)
            inputs = processor(images=img, return_tensors="pt")
            # Make prediction
            outputs = model(**inputs)
            logits = outputs.logits
            # Get the predicted class
            predicted_class = logits.argmax().item()
            return predicted_class


@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/index')
def detect():
    return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred_index = prediction(file_path)
        class_name = class_names.get(pred_index, "Unknown")
        description =disease_info['description'][pred_index]
        prevent = disease_info['Possible Steps'][pred_index]
        image_url = disease_info['image_url'][pred_index]
        supplement_name = supplement_info['supplement name'][pred_index]
        supplement_image_url = supplement_info['supplement image'][pred_index]
        supplement_buy_link = supplement_info['buy link'][pred_index]
        return render_template('submit.html',class_name=class_name, desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred_index ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)
    return render_template('index.html')


@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']),
                           disease=list(disease_info['disease_name']), buy=list(supplement_info['buy link']),
                           class_keys=list(class_names.keys()))


if __name__ == '__main__':
    app.run(debug=True)
