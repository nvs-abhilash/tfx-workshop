import os
from typing import List
import absl
import tensorflow as tf
from tfx import v1 as tfx
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.components.trainer.fn_args_utils import FnArgs
import tensorflow_transform as tft
from tfx_bsl.tfxio import dataset_options

_TRAIN_DATA_SIZE = 128
_EVAL_DATA_SIZE = 128
_TRAIN_BATCH_SIZE = 32
_EVAL_BATCH_SIZE = 32
_CLASSIFIER_LEARNING_RATE = 1e-3
_FINETUNE_LEARNING_RATE = 7e-6
_CLASSIFIER_EPOCHS = 30

_IMAGE_KEY = 'image'
_LABEL_KEY = 'label'


def _image_augmentation(image_features):
    """Perform image augmentation on batches of images .

    Args:
    image_features: a batch of image features

    Returns:
    The augmented image features
    """
    batch_size = tf.shape(image_features)[0]
    image_features = tf.image.random_flip_left_right(image_features)
    image_features = tf.image.resize_with_crop_or_pad(image_features, 250, 250)
    image_features = tf.image.random_crop(image_features,
                                          (batch_size, 224, 224, 3))
    return image_features


def _data_augmentation(feature_dict):
    """Perform data augmentation on batches of data.

    Args:
    feature_dict: a dict containing features of samples

    Returns:
    The feature dict with augmented features
    """
    image_features = feature_dict[_transformed_name(_IMAGE_KEY)]
    image_features = _image_augmentation(image_features)
    feature_dict[_transformed_name(_IMAGE_KEY)] = image_features
    return feature_dict


def _input_fn(file_pattern: List[str],
              data_accessor: DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              is_train: bool = False,
              batch_size: int = 200) -> tf.data.Dataset:
    """Generates features and label for tuning/training.

    Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A TFTransformOutput.
    is_train: Whether the input dataset is train split or not.
    batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch

    Returns:
    A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=_transformed_name(_LABEL_KEY)),
        tf_transform_output.transformed_metadata.schema)
    # Apply data augmentation. We have to do data augmentation here because
    # we need to apply data agumentation on-the-fly during training. If we put
    # it in Transform, it will only be applied once on the whole dataset, which
    # will lose the point of data augmentation.
    if is_train:
        dataset = dataset.map(lambda x, y: (_data_augmentation(x), y))

    return dataset


def _transformed_name(key):
    return key + '_xf'


def _freeze_model_by_percentage(model: tf.keras.Model, percentage: float):
    """Freeze part of the model based on specified percentage.

  Args:
    model: The keras model need to be partially frozen
    percentage: the percentage of layers to freeze

  Raises:
    ValueError: Invalid values.
  """
    if percentage < 0 or percentage > 1:
        raise ValueError('Freeze percentage should between 0.0 and 1.0')

    if not model.trainable:
        raise ValueError(
            'The model is not trainable, please set model.trainable to True')

    num_layers = len(model.layers)
    num_layers_to_freeze = int(num_layers * percentage)
    for idx, layer in enumerate(model.layers):
        if idx < num_layers_to_freeze:
            layer.trainable = False
        else:
            layer.trainable = True


def _build_keras_model() -> tf.keras.Model:
    """Creates a Image classification model with MobileNet backbone.

    Returns:
        The image classifcation Keras Model and the backbone MobileNet model
    """
    # We create a MobileNet model with weights pre-trained on ImageNet.
    # We remove the top classification layer of the MobileNet, which was
    # used for classifying ImageNet objects. We will add our own classification
    # layer for CIFAR10 later. We use average pooling at the last convolution
    # layer to get a 1D vector for classifcation, which is consistent with the
    # origin MobileNet setup
    base_model = tf.keras.applications.MobileNet(input_shape=(224, 224, 3),
                                                 include_top=False,
                                                 weights='imagenet',
                                                 pooling='avg')
    base_model.input_spec = None

    # We add a Dropout layer at the top of MobileNet backbone we just created to
    # prevent overfiting, and then a Dense layer to classifying CIFAR10 objects
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3),
                                   name=_transformed_name(_IMAGE_KEY)),
        base_model,
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Freeze the whole MobileNet backbone to first train the top classifer only
    _freeze_model_by_percentage(base_model, 1.0)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(lr=_CLASSIFIER_LEARNING_RATE),
        metrics=['sparse_categorical_accuracy'])
    model.summary(print_fn=absl.logging.info)

    return model


def run_fn(fn_args: tfx.components.FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files,
                              fn_args.data_accessor,
                              tf_transform_output,
                              is_train=True,
                              batch_size=_TRAIN_BATCH_SIZE)

    eval_dataset = _input_fn(fn_args.eval_files,
                             fn_args.data_accessor,
                             tf_transform_output,
                             is_train=False,
                             batch_size=_EVAL_BATCH_SIZE)

    model = _build_keras_model()
    steps_per_epoch = int(_TRAIN_DATA_SIZE / _TRAIN_BATCH_SIZE)
    total_epochs = int(fn_args.train_steps / steps_per_epoch)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq='batch')
    model.fit(train_dataset,
              epochs=total_epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=eval_dataset,
              validation_steps=fn_args.eval_steps,
              callbacks=[tensorboard_callback])

    model.save(fn_args.serving_model_dir, save_format='tf')
