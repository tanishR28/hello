def call_model_function(images, call_model_args=None, expected_keys=None):
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    target = call_model_args["target_class"]
    with tf.GradientTape() as tape:
        tape.watch(images)                                  # track gradients w.r.t. the pixels
        output = model(images, training=False)[:, target]   # score of the target class
    gradients = tape.gradient(output, images).numpy()       # d score / d pixels
    return {saliency.INPUT_OUTPUT_GRADIENTS: gradients}     # XRAI expects exactly this key
