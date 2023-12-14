import io

from flask import Flask, request, jsonify, abort,render_template

from flaskext.mysql import MySQL

import os

from datetime import date

from datetime import datetime

import pytz

import mysql.connector

import tensorflow as tf

os.environ["KERAS_BACKEND"] = "tensorflow"  # @param ["tensorflow", "jax", "torch"]

from tensorflow import data as tf_data
import tensorflow_datasets as tfds
import keras
import keras_cv
import numpy as np
from keras_cv import bounding_box
import os
from keras_cv import visualization
import tqdm
from PIL import Image

#pretrained model == model that already I trained with the custom data
pretrained_model = keras_cv.models.YOLOV8Detector.from_preset(
    "yolo_v8_m_pascalvoc", bounding_box_format="xywh"
)

# Initialize Flask application
app = Flask(__name__)

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="parking"
)

mycursor = connection.cursor()






# Render HTML page for the root URL
@app.route('/')
def index():
    return "<h1>Meeting Room Detection</h1>"

@app.route('/detections', methods=['POST'])
def predict():
    data = {"success": False}
    if request.method == "POST":
        if request.files.get("image"):
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = image.resize((640, 640))

            # Convert PIL Image to NumPy array
            image_array = np.array(image)
            image_array = np.expand_dims(image_array, axis=0)

            # Perform prediction
            result_value = pretrained_model.predict(image_array)
            testing = result_value["classes"]
            human = np.count_nonzero(testing == 14)

            today = date.today()

            timezone = pytz.timezone("Asia/Jakarta")
            hour = datetime.now(timezone).hour

            sql_syntax = "INSERT INTO parkingCount (person_sum, date, hour) VALUES (%s, %s, %s)"
            val = (human, today, hour)

            mycursor.execute(sql_syntax, val)
            connection.commit()


            # Return the count of the specific class as JSON
            return jsonify({"human_count": human})

    # If the image processing fails or there's no image in the request, return an error message
    return jsonify({"error": "No image or image processing failed."})
if __name__ == '__main__':
    app.run()