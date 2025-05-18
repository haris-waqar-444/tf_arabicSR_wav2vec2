import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Layer

# Define the custom layer class with the same name and package as in test.py
@tf.keras.saving.register_keras_serializable(package="CustomLayers")
class Wav2Vec2Layer(Layer):
    def __init__(self, **kwargs):
        super(Wav2Vec2Layer, self).__init__(**kwargs)
        # AUDIO_MAXLEN is defined in test.py, we need to define it here too
        AUDIO_MAXLEN = 246000
        self.pretrained_layer = hub.KerasLayer("https://tfhub.dev/vasudevgupta7/wav2vec2/1", trainable=True, input_shape=(AUDIO_MAXLEN,))

    def call(self, inputs):
        return self.pretrained_layer(inputs)

    def get_config(self):
        config = super(Wav2Vec2Layer, self).get_config()
        return config

# Load the .keras model with custom_objects
model = tf.keras.models.load_model('finetuned-wav2vec2.keras')

# Convert the Keras model to a TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open('ASR_wav2vec2.tflite', 'wb') as f:
    f.write(tflite_model)

print("Conversion to TFLite completed!")
