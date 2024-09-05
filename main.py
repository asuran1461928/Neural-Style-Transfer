import os
import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

# Set environment for TensorFlow Hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

# Define functions
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    st.pyplot(plt)

# Streamlit app
st.title('Neural Style Transfer')

# Upload images
uploaded_content_image = st.file_uploader("Upload Content Image", type=['jpg', 'png'])
uploaded_style_image = st.file_uploader("Upload Style Image", type=['jpg', 'png'])

if uploaded_content_image is not None and uploaded_style_image is not None:
    # Save uploaded files
    content_image_path = 'content_image.jpg'
    style_image_path = 'style_image.jpg'
    
    with open(content_image_path, 'wb') as f:
        f.write(uploaded_content_image.getvalue())
    
    with open(style_image_path, 'wb') as f:
        f.write(uploaded_style_image.getvalue())

    # Load images
    content_image = load_img(content_image_path)
    style_image = load_img(style_image_path)

    # Display images
    st.subheader("Content Image")
    imshow(content_image, 'Content Image')
    
    st.subheader("Style Image")
    imshow(style_image, 'Style Image')

    # Load TensorFlow Hub model
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    # Stylize the image
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

    # Display stylized image
    st.subheader("Stylized Image")
    imshow(stylized_image, 'Stylized Image')
else:
    st.text("Please upload both content and style images.")
