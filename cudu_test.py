# This is a sample Python script.
import tensorflow as tf



def len_gpu_available():

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


def test_tensor_assignment():
    tf.debugging.set_log_device_placement(True)
    print("creating tensors, are we using gpu")
    # Create some tensors
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

    print(c)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    len_gpu_available()
    test_tensor_assignment()
    print("we have an 8.6 compute capability baby")