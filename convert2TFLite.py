import tensorflow as tf

saved_critic = '/mnt/96a66be0-609e-43bd-a076-253e3c725b17/Python/RL testing/save_models/TTT2D_Critic'
saved_actor = '/mnt/96a66be0-609e-43bd-a076-253e3c725b17/Python/RL testing/save_models/TTT2D_actor'
# Convert the model

converter = tf.lite.TFLiteConverter.from_saved_model(saved_critic) # path to the SavedModel directory
tflite_model = converter.convert()
# Save the model.

with open('citic_model.tflite', 'wb') as f:

  f.write(tflite_model)
  
converter = tf.lite.TFLiteConverter.from_saved_model(saved_actor) # path to the SavedModel directory
tflite_model = converter.convert()
# Save the model.

with open('actor_model.tflite', 'wb') as f:

  f.write(tflite_model)
