import tensorflow as tf

_IMAGE_KEY = 'image'
_LABEL_KEY = 'label'


def _transformed_name(key):
    return key + '_xf'


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
    outputs = {}

    # tf.io.decode_png function cannot be applied on a batch of data.
    # We have to use tf.map_fn
    image_features = tf.map_fn(lambda x: tf.io.decode_png(x[0], channels=3),
                               inputs[_IMAGE_KEY],
                               dtype=tf.uint8)
    image_features = tf.image.resize(image_features, [224, 224])
    image_features = tf.keras.applications.mobilenet.preprocess_input(
        image_features)

    outputs[_transformed_name(_IMAGE_KEY)] = image_features
    # TODO(b/157064428): Support label transformation for Keras.
    # Do not apply label transformation as it will result in wrong evaluation.
    outputs[_transformed_name(_LABEL_KEY)] = inputs[_LABEL_KEY]

    return outputs
