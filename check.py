import tensorflow as tf

model = tf.keras.models.load_model("./data/petbreed_model")
model.summary()