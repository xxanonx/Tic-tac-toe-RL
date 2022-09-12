import tensorflow as tf

saved_critic = 'save_models/TTT2D_Critic'
saved_actor = 'save_models/TTT2D_actor'

# Convert the model

converter = tf.lite.TFLiteConverter.from_saved_model(saved_critic) # path to the SavedModel directory
tflite_model = converter.convert()
# Save the model.

with open('save_models/citic_modelM1.tflite', 'wb') as f:

  f.write(tflite_model)
  
converter = tf.lite.TFLiteConverter.from_saved_model(saved_actor) # path to the SavedModel directory
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]

tflite_model = converter.convert()
# Save the model.

with open('save_models/actor_modelM1.tflite', 'wb') as f:

  f.write(tflite_model)
