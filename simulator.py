import numpy as np
from tqdm import tqdm
from random import random, randint, uniform, sample
from PIL import Image, ImageDraw
from opensimplex import OpenSimplex

MAX_INT = int(1e8)
IMAGE_WIDTH, IMAGE_HEIGHT = 240, 180
FIRST_FREQUENCY_MEAN = 0.1
FIRST_FREQUENCY_STD = 0.03
SECOND_FREQUENCY_MEAN = 0.03
SECOND_FREQUENCY_STD = 0.006
THIRD_FREQUENCY_MEAN = 0.01
THIRD_FREQUENCY_STD = 0.002
FIRST_FREQUENCY_WEIGHT_MEAN = 0.3
FIRST_FREQUENCY_WEIGHT_STD = 0.1
SECOND_FREQUENCY_WEIGHT_MEAN = 0.3
SECOND_FREQUENCY_WEIGHT_STD = 0.1
THIRD_FREQUENCY_WEIGHT_MEAN = 0.3
THIRD_FREQUENCY_WEIGHT_STD = 0.1

BORDER_LINE_MEAN = 0.2916
BORDER_LINE_STD = 0.01
BORDER_OFFSET = 0.2
TOP_THRESHOLD_MEAN = -0.1
TOP_THRESHOLD_STD = 0.1
BOT_THRESHOLD_MEAN = 0.8
BOT_THRESHOLD_STD = 0.1

BALL_PLACEMENT_SLOPE_MEAN = 6.5
BALL_PLACEMENT_SLOPE_STD = 0.1
BALL_PLACEMENT_OFFSET_MEAN = 0.2
BALL_PLACEMENT_OFFSET_STD = 0.01
BALL_RADIUS_NOISE_MEAN = 0.0
BALL_RADIUS_NOISE_STD = 0.004

BALL_NOISE_NUM_ARCS = 360
BALL_NOISE_SMOOTH_EPSILON = 1.0
BALL_BORDER_NOISE_MEAN = 0.2
BALL_BORDER_NOISE_STD = 0.1

BALL_COEFFICIENT_MEAN = 5.0
BALL_COEFFICIENT_STD = 0.1

def get_number_of_balls():
  r = random()
  if r < 0.2:
    return 0
  if r < 0.45:
    return 1
  if r < 0.7:
    return 2
  if r < 0.95:
    return 3
  if r < 0.98:
    return 4
  if r < 0.99:
    return 5
  else:
    return 6

def sample_noise(x_size, y_size, freq):
  tmp = OpenSimplex(seed=randint(0, MAX_INT))
  noise = np.zeros((x_size, y_size))
  for i in xrange(x_size):
    for j in xrange(y_size):
      noise[i, j] = tmp.noise2d(i*freq, j*freq)
  return noise

def display_noise(noise):
  noise = noise.copy()
  noise += 1
  noise *= 255
  return Image.fromarray(noise.astype(np.uint8))
  
def display_mask(mask):
  return Image.fromarray((mask*255).astype(np.uint8))

def combine_noises(noises, weights):
  weights = np.array(weights)
  weights /= np.sum(weights)
  weights = list(weights)
  ret_noise = np.zeros_like(noises[0])
  for noise, weight in zip(noises, weights):
    ret_noise += weight * noise
  return ret_noise

def get_position(center, radius):
  return  (center[0] - radius, 
           center[1] - radius, 
           center[0] + radius,
           center[1] + radius)

def get_noisy_circle_mask(center, radius, noise_scale=0.5):
  image = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT))
  draw = ImageDraw.Draw(image)
  res = 0
  noisy_radius = radius
  deltas = np.zeros((BALL_NOISE_NUM_ARCS))
  for i in range(BALL_NOISE_NUM_ARCS):
    deltas[i] = uniform(-1, 1) * noise_scale
  deltas -= np.mean(deltas)
  
  for i in range(BALL_NOISE_NUM_ARCS):
    noisy_radius += deltas[i]
    draw.pieslice(get_position(center, noisy_radius),
                  start=res, 
                  end=res + 360/BALL_NOISE_NUM_ARCS \
                  + BALL_NOISE_SMOOTH_EPSILON,
                  fill=255)
    res += 360 / BALL_NOISE_NUM_ARCS 
  return np.array(image) / 255.0

class Simulator(object):
  
  def populate_noise_groups(self, amount):
    print "Populating Noise Groups..."
    self.first_group = []
    self.second_group = []
    self.third_group = []
    for i in tqdm(range(amount), ncols=100):
      # sample frequencies
      first_frequency = np.random.normal(FIRST_FREQUENCY_MEAN, FIRST_FREQUENCY_STD)
      second_frequency = np.random.normal(SECOND_FREQUENCY_MEAN, SECOND_FREQUENCY_STD)
      third_frequency = np.random.normal(THIRD_FREQUENCY_MEAN, THIRD_FREQUENCY_STD)
      # sample noises
      first_noise = sample_noise(IMAGE_HEIGHT, IMAGE_WIDTH, first_frequency)
      second_noise = sample_noise(IMAGE_HEIGHT, IMAGE_WIDTH, second_frequency)
      third_noise = sample_noise(IMAGE_HEIGHT, IMAGE_WIDTH, third_frequency)
      self.first_group.append(first_noise)
      self.second_group.append(first_noise)
      self.third_group.append(first_noise)
    
  def sample_noises(self):
    first_noise = sample(self.first_group, 1)[0]
    second_noise = sample(self.second_group, 1)[0]
    third_noise = sample(self.second_group, 1)[0]
    return [first_noise, second_noise, third_noise] 
    
  def generate(self):
    noises = self.sample_noises()
    first_weight = np.random.normal(FIRST_FREQUENCY_WEIGHT_MEAN, FIRST_FREQUENCY_WEIGHT_STD)
    second_weight = np.random.normal(SECOND_FREQUENCY_WEIGHT_MEAN, SECOND_FREQUENCY_WEIGHT_STD)
    third_weight = np.random.normal(THIRD_FREQUENCY_WEIGHT_MEAN, THIRD_FREQUENCY_WEIGHT_STD)
    weights = [first_weight, second_weight, third_weight]
    combined_noise = combine_noises(noises, weights)
    # Determine border line and thresholds
    border_line = np.abs(np.random.normal(BORDER_LINE_MEAN, BORDER_LINE_STD))
    top_threshold = np.random.normal(TOP_THRESHOLD_MEAN, TOP_THRESHOLD_STD)
    bot_threshold = np.random.normal(BOT_THRESHOLD_MEAN, BOT_THRESHOLD_STD)
    top_height = int(border_line * IMAGE_HEIGHT)
    bot_height = IMAGE_HEIGHT - top_height
    top_threshold_matrix = np.ones((top_height, IMAGE_WIDTH)) * top_threshold
    bot_threshold_matrix = np.ones((bot_height, IMAGE_WIDTH)) * bot_threshold
    threshold_matrix = np.vstack([top_threshold_matrix, bot_threshold_matrix])
    # Place the balls randomly
    ball_placement_slope = np.random.normal(BALL_PLACEMENT_SLOPE_MEAN, BALL_PLACEMENT_SLOPE_STD)
    ball_placement_offset = np.random.normal(BALL_PLACEMENT_OFFSET_MEAN, BALL_PLACEMENT_OFFSET_STD)
    n_balls = get_number_of_balls()
    label = (0, 0, 0)
    min_distance = MAX_INT
    for _ in range(n_balls):
      ball_x = np.random.uniform(0, IMAGE_WIDTH)
      ball_y = np.random.uniform((border_line + BORDER_OFFSET) * IMAGE_HEIGHT, IMAGE_HEIGHT)
      radius_noise = np.random.normal(BALL_RADIUS_NOISE_MEAN, BALL_RADIUS_NOISE_STD)
      radius = (ball_y - ball_placement_offset * IMAGE_HEIGHT) / ball_placement_slope 
      radius += radius_noise * IMAGE_HEIGHT
      radius = abs(radius) # just in case it goes negative
      ball_border_noise = np.random.normal(BALL_BORDER_NOISE_MEAN, BALL_BORDER_NOISE_STD)
      ball_mask = get_noisy_circle_mask((ball_x, ball_y), 
                                      radius, 
                                      noise_scale=ball_border_noise)
      ball_coefficient = np.random.normal(BALL_COEFFICIENT_MEAN, BALL_COEFFICIENT_STD)
      combined_noise += ball_mask * ball_coefficient
      distance = (IMAGE_WIDTH / 2 - ball_x) ** 2 + (IMAGE_HEIGHT - ball_y) ** 2
      if distance < min_distance:
        min_distance = distance
        label = (ball_y, ball_x, radius)
    return combined_noise > threshold_matrix, label
    
