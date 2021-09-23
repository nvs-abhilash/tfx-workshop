# Model Serving

## Tensorflow Serving

1. Pull the docker image
    ```bash
    docker pull tensorflow/serving
    ```

2. Download the trained model and put the `cifar-tfs-saved_model` inside a folder called `saved_models`
    ```bash
    mkdir tfx-workshop
    cd tfx-workshop
    mkdir saved_models
    cd saved_models
    # put cifar-tfs-saved_model folder here
    ```

3. Run the Tensorflow Serving container

    ```bash
    docker container run -d --rm -p 8501:8501 --mount type=bind,source=$(pwd)\saved_models,target=/models -e MODEL_NAME=cifar-tfs-saved_model tensorflow/serving
    ```

4. Test the published API using the helper client script `tf_serving_client.py`

    ```bash
    conda create -n tfxworkshop python=3.7
    conda activate tfxworkshop
    python -m pip install -r tfserving/requirements.txt
    python tfserving/tf_serving_client.py
    ```


## Deploying in Flask 

1. Test the flask server and client locally.

    Terminal 1:
    ```bash
    cd model_serving
    conda activate tfxworkshop
    python -m pip install -r flask/requirements.txt
    python flask/flask_server.py
    ```

    Terminal 2:
    ```bash
    cd model_serving
    conda activate tfxworkshop
    python flask/flask_client.py
    ```

    Validate whether the output is matching with tensorflow serving client's output or not.


2. Build your flask Docker image and run the container

    Terminal 1:
    ```bash
    cd model_serving/flask
    docker image build -t tfxworkshop_flask .
    docker container run -d -p 5000:5000 tfxworkshop_flask
    ```

    Terminal 2:
    ```bash
    cd model_serving
    conda activate tfxworkshop
    python flask/flask_client.py
    ```
