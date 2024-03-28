# 1. Why would you want to use the tf.data API?
# tf.data has many useful features, like handling data that doesn't fit in memory, using multiple cores,
# multithreading etc

# 2. What are the benefits of splitting a large dataset into multiple files?
# It makes shuffling easier. The files can be read at the same time and interleaved.

# 3. During training, how can you tell that your input pipeline is the bottleneck? What can you do to fix it?

# 4. Can you save any binary data to a TFRecord file, or only serialized protocol buffers?
# Yes you can save any binary data, but people tend to use protobufs.

# 5. Why would you go through the hassle of converting all your data to the Example protobuf format? Why not use your
# own protobuf definition? Protobufs are portable, extensible and efficient?

# 6. When using TFRecords, when would you want to activate compression? Why not do it systematically? Use compression
# when data is remote so it downloads quicker. Using it systematically is probably bad because compression and
# decompression take time.

# 7. Data can be preprocessed directly when writing the data files, or within the tf.data pipeline, or in preprocessing
# layers within your model. Can you list a few pros and cons of each option?
# Preprocessing layers within your model is good because reduces the likelihood preproessing mismatch, it makes training
# slower though.
# Doing it when writing the files is good because it's simple. tf.data pipelines are very powerful for
# doing complex stuff.

# 8. Name a few common ways you can encode categorical integer features. What about text?
# Categorical integer features can be one-hot encoded or multi-hot encoded.
# Text can use a StringLookup layer.
# This inspects text and takes each word to be a separate categorical feature which can then be one-hot encoded.
# For text embeddings can also be used.
import tensorflow
from tensorflow.train import Example, Feature, Features, BytesList, Int64List

BATCH_SIZE = 100


def q9a():
    (data_train_full, labels_train_full), (test_data, test_labels) = tensorflow.keras.datasets.fashion_mnist.load_data()
    data_valid, data = data_train_full[:5000], data_train_full[5000:]
    valid_labels, labels = labels_train_full[:5000], labels_train_full[5000:]
    dataset = tensorflow.data.Dataset.from_tensor_slices((data, labels))
    valid_set = tensorflow.data.Dataset.from_tensor_slices((data_valid, valid_labels))
    test_set = tensorflow.data.Dataset.from_tensor_slices((test_data, test_labels))
    dataset = dataset.shuffle(buffer_size=55000, seed=0).batch(BATCH_SIZE)
    for i, batch in enumerate(dataset):
        with tensorflow.io.TFRecordWriter(f"datasets/fashion_mnist/batch_{i:05}.tfrecord") as f:
            for j in range(BATCH_SIZE):
                image = batch[0][j]
                label = batch[1][j]
                f.write(create_example(image, label).SerializeToString())


def create_example(image, label):
    image_data = tensorflow.io.serialize_tensor(image)
    return Example(
        features=Features(
            feature={
                "image": Feature(bytes_list=BytesList(value=[image_data.numpy()])),
                "label": Feature(int64_list=Int64List(value=[label])),
            }))


def q9b():
    # Finally, use a Keras model to train these datasets, including a preprocessing layer to standardize each input
    # feature. Try to make the input pipeline as efficient as possible, using TensorBoard to visualize profiling data.
    pass


