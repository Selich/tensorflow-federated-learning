from __future__ import absolute_import, division, print_function
import collections
import warnings
import numpy as np
import six
from six.moves import range
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_federated as tff

warnings.simplefilter('ignore')

tf.compat.v1.enable_v2_behavior()

np.random.seed(0)

if six.PY3:
    tff.framework.set_default_executor(tff.framework.create_local_executor())

tff.federated_computation(lambda: 'Hello')()

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

dataset = emnist_train.create_tf_dataset_for_client( emnist_train.client_ids[0] )

element = iter(dataset).next()
element['label'].numpy()

NUM_CLIENTS = 10
NUM_EPOCHS = 10
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500

def preprocess(dataset):

    def element_fun(element):
        return collections.OrderedDict([
            ('x', tf.reshape(element['pixels'], [-1])),
            ('y', tf.reshape(element['label'], [1])),
    ])

    return dataset.repeat(NUM_EPOCHS).map(element_fun).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)

preprocessed_dataset = preprocess(dataset)
batch = tf.nest.map.structure(
        lambda x: x.numpy(), iter(preprocessed_dataset).next()
        )

batch





