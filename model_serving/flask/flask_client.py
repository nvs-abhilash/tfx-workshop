import json
import requests
from requests.api import head
import cv2
import numpy as np

def main():
    labels = [
        "airplane",
        "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship",
        "truck"
    ]

    img = cv2.cvtColor(cv2.imread("sample_airplane.jpg"), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    data = json.dumps({"instances": img[None, ...].tolist()})
    headers = {"content-type": "application/json"}

    response = requests.post(
        'http://localhost:5000/predict',
        data=data,
        headers=headers)

    print(response.json())
    result = int(np.argmax(response.json()['predictions'][0]))
    print(labels[result])


if __name__ == '__main__':
    main()
