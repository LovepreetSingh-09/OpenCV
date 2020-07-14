from tensorflow.keras.applications import  xception
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
import tensorflow as tf

app = flask.Flask(__name__)

def load_model():
    global model
    model = Xception(weights="imagenet")


def preprocessing_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = xception.preprocess_input(image)
    return image


@app.route("/predict", methods=["POST"])
def predict():
    result = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = preprocessing_image(image, target=(224, 224))
            predictions = model.predict(image)
            results = imagenet_utils.decode_predictions(predictions)
            result["predictions"] = []
            for (imagenet_id, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                result["predictions"].append(r)
            result["success"] = True
    return flask.jsonify(result)


@app.route("/")
def home():
    result = {"success": True}
    return flask.jsonify(result)


if __name__ == "__main__":
    print("Loading Keras pre-trained model")
    load_model()
    print("Starting")
    app.run()
