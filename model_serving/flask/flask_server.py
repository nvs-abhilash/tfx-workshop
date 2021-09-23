from flask import Flask, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model("saved_models/cifar10-tfx/1632361176")


@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    imgs = np.asarray(content["instances"])
    preds = model.predict(imgs)
    print(preds)

    return {"predictions": preds.tolist()}


if __name__ == '__main__':
    app.run()
