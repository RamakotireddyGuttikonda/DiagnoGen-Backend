import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
H = W = 256
def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W,H)) / 255.0
    return x.astype(np.float32)

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x,(W,H))/255.0
    x = np.expand_dims(x, axis=-1)
    return x.astype(np.float32)

def tf_parse(x, y):
    def _parse(x, y):
        return read_image(x), read_mask(y)
    x, y = tf.numpy_function(_parse, [x,y], [tf.float32, tf.float32])
    x.set_shape([H,W,3])
    y.set_shape([H,W,1])
    return x, y

def tf_dataset(X, Y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((X,Y))
    dataset = dataset.map(tf_parse).batch(batch).prefetch(tf.data.AUTOTUNE)
    return dataset

def load_dataset(path, split=0.2):
    images = sorted(glob(os.path.join(path,"images","*.png")))
    masks = sorted(glob(os.path.join(path,"masks","*.png")))
    train_x, test_x, train_y, test_y = train_test_split(images, masks, test_size=split, random_state=42)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=split, random_state=42)
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)