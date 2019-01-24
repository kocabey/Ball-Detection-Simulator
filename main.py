import numpy as np
import tensorflow as tf
from time import time
from tqdm import tqdm
from network import Network
from simulator import Simulator, display_mask

BATCH_SIZE = 64
EPOCH_SIZE = 100
NOISE_AMOUNT = 10
IMAGE_WIDTH, IMAGE_HEIGHT = 240, 180
SAVE_PATH = "/home/ubuntu/Code/Domain-Randomization/Models/ball_model"

network = Network()
simulator = Simulator()
simulator.populate_noise_groups(NOISE_AMOUNT)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def get_batch(size):
  image_batch = np.zeros((size, IMAGE_HEIGHT, IMAGE_WIDTH, 1))
  label_batch = np.zeros((size, 3))
  for i in range(size):
    image, label = simulator.generate()
    image_batch[i] = np.array(image).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    label_batch[i] = np.array(label)
  return image_batch, label_batch

while True:
  mean_loss = 0
  for _ in tqdm(range(EPOCH_SIZE), ncols=100):
    image_batch, label_batch = get_batch(BATCH_SIZE)
    loss, _ = sess.run([network.loss, network.train_step],
                       feed_dict={
                         network.image_: image_batch,
                         network.label_: label_batch
                       })
    mean_loss += loss
  print mean_loss / EPOCH_SIZE
  simulator.populate_noise_groups(NOISE_AMOUNT)
  network.saver.save(sess, SAVE_PATH)
