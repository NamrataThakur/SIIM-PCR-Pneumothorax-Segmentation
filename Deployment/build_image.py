
# coding: utf-8

# In[ ]:


import tensorflow as tf


# In[ ]:


def read_image(path):
    
    x = tf.io.read_file(path)
    x = tf.image.decode_png(x, channels=3)
    x = tf.image.convert_image_dtype(x, tf.float32)

    return x

def classify_decode(path):
    file_bytes = tf.io.read_file(path)
    img = tf.image.decode_png(file_bytes, channels=3)
    img = tf.cast(img, tf.float32) / 255.0

    return img

