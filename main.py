from flask import Flask, request, render_template
import os
import numpy as np
import keras
import tensorflow as tf
import random
import string

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)
skin_model = keras.models.load_model("model_e100")
OUTPUT_DIR = 'static'
dicto = {
    0: 'Melanocytic nevi',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Vascular lesions',
    4: 'Dermatofibroma',
    5: 'Melanoma',
    6: 'Actinic keratoses',
    7: 'No disease'
}


def generate_filename():
    return ''.join(random.choices(string.ascii_lowercase, k=20)) + '.jpg'


def get_prediction(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    # img = image.load_img(image_path, target_size=(224, 224))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    y_predict = skin_model.predict(image)
    prediction = np.argmax(y_predict)
    class_name = prediction
    return class_name


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        uploaded_file = request.files["file"]
        if uploaded_file:
            image_path = os.path.join(OUTPUT_DIR, generate_filename())
            uploaded_file.save(image_path)
            class_name = get_prediction(image_path)
            class_name = dicto[class_name]
            if class_name == 'Actinic keratoses':

                result = {
                    'class_name': class_name,
                    'path_to_image': image_path}
                return render_template("Actinic Keratosis.html", result=result)
            elif class_name == 'Melanoma':

                result = {
                    'class_name': class_name,
                    'path_to_image': image_path}
                return render_template("Melanoma.html", result=result)
            elif class_name == 'Benign keratosis-like lesions':

                result = {
                    'class_name': class_name,
                    'path_to_image': image_path}
                return render_template("Benign Keratosisa.html", result=result)
            elif class_name == 'Melanocytic nevi':

                result = {
                    'class_name': class_name,
                    'path_to_image': image_path}
                return render_template("Melanocytic.html", result=result)
            elif class_name == 'Vascular lesions':

                result = {
                    'class_name': class_name,
                    'path_to_image': image_path}
                return render_template("Vascular lesions.html", result=result)
            elif class_name == 'Dermatofibroma':

                result = {
                    'class_name': class_name,
                    'path_to_image': image_path}
                return render_template("Dermatofibroma.html", result=result)
            elif class_name == 'Basal cell carcinoma':
                result = {
                    'class_name': class_name,
                    'path_to_image': image_path}
                return render_template("Bascal Cell Carcinoma.html", result=result)
            elif class_name == 'No disease':
                result = {
                    'class_name': class_name,
                    'path_to_image': image_path}
                return render_template("No.html", result=result)

            return 0

        else:
            return render_template("first_page.html", im="No Image Found")
    return render_template("first_page.html", im="")


if __name__ == '__main__':
    app.run(debug=True)
