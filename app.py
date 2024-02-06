from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)

app = Flask(__name__)

# Load the model
model = load_model('/Project/cnn_model_2.h5')

def model_predict(img_path, model):
    test_image = image.load_img(img_path, target_size=(256, 256))  # Updated size
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    return result


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the uploaded image file
    image_file = request.files['file']

    # Save the file to the uploads folder
    basepath = os.path.dirname(os.path.realpath('__file__'))
    file_path = os.path.join(basepath, '/Project/uploads/', secure_filename(image_file.filename))
    image_file.save(file_path)

    try:
        # Make prediction
        result = model_predict(file_path, model)

        # Define your categories
        categories = ['Tomato_Bacterial_spot Disease', 'Tomato_Early_blight Disease', 'Tomato_Late_blight Disease', 'Tomato_Leaf_Mold Disease' , 'Tomato_Septoria_leaf_spot Disease' , 'Tomato_Spider_mites_Two_spotted_spider_mite Disease' , 'Tomato__Target_Spot Disease' , 'Tomato__Tomato_YellowLeaf__Curl_Virus' , 'Tomato__Tomato_mosaic_virus' , 'Tomato_healthy Leaf' , 'Unknown Crop']  # Update with your categories

        # Find the index of the class with the highest probability
        pred_class_index = np.argmax(result)

        # Map the index to the corresponding category
        pred_category = categories[pred_class_index]

        # Return the prediction category
        return pred_category
    except Exception as e:
        return str(e)  # Return the error message as a response


if __name__ == '__main__':
   app.run()