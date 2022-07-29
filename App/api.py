import os
from os import path
import cv2  # opencv
import shutil
import numpy as np
import time
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing import image
from skimage import io
import numpy as np
from PIL import Image

import tensorflow as tf
import tensorflow.keras.backend as K

from flask import Flask, request
from flask import render_template

app = Flask(__name__)
UPLOAD_FOLDER = UPLOAD_FOLDER = "E:/Misc/B9-Project/App/static/uploads"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

classificationModel = load_model("detection/classification_model.h5")

segmentationModel = model_from_json(
    open("segmentation/ResNext50-seg-model.json").read()
)
segmentationModel.load_weights(
    os.path.join(
        os.path.dirname("segmentation/ResNext50-seg-model.json"), "seg_model.h5"
    )
)


def prediction(test_image, model, model_seg, fileName):

    result = {}
    # preprocessing
    img = io.imread(test_image)
    img = img * (1.0 / 255)
    img = cv2.resize(img, (256, 256))
    img = np.array(img, dtype=np.float64)
    img = np.reshape(img, (1, 256, 256, 3))

    class_pred = model.predict(img)  # classification
    print(class_pred)
    if np.argmax(class_pred) == 0:
        result["classification_label"] = "Tumor Not Present"
    else:
        result["classification_label"] = "Tumor Present"
    result["classification_prob"] = class_pred[0][np.argmax(class_pred)] * 100

    X = np.empty((1, 256, 256, 3))
    img = io.imread(test_image)
    img = cv2.resize(img, (256, 256))
    img = np.array(img, dtype=np.float64)

    img -= img.mean()
    img /= img.std()
    X[
        0,
    ] = img

    mask_pred = model_seg.predict(X)  # segmentation

    result["seg_mask"] = mask_pred
    pred = np.array(mask_pred).squeeze().round()
    img_ = io.imread(test_image)
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    img_[pred == 1] = (0, 255, 150)
    im = Image.fromarray(img_)
    im.save(fileName)
    print(fileName)
    shutil.move(
        "E:/Misc/B9-Project/App/" + fileName,
        "E:/Misc/B9-Project/App/static/output/" + fileName,
    )
    return result


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["inputImage"]
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)
            preds = prediction(
                image_location,
                classificationModel,
                segmentationModel,
                image_file.filename,
            )
            return render_template(
                "index.html",
                pred_prob=preds["classification_prob"],
                pred=preds["classification_label"],
                imageInput=image_file.filename,
                imageOutput=image_file.filename,
            )
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
