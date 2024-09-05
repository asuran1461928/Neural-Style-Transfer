import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Helper function to load and preprocess image
def load_and_preprocess_image(image_path, max_dim=512):
    img = Image.open(image_path)
    img = np.array(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# Load VGG19 model and select the layers for style and content
def get_model():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    outputs = [vgg.get_layer(name).output for name in (style_layers + content_layers)]
    
    return tf.keras.Model([vgg.input], outputs), style_layers, content_layers

# Define the style and content losses
def compute_loss(model, style_outputs, content_outputs, style_weight=1e-2, content_weight=1e4):
    style_weight_per_layer = 1.0 / len(style_outputs)
    style_loss = tf.add_n([style_weight_per_layer * tf.reduce_mean((style_outputs[name] - model.outputs[name]) ** 2) for name in style_outputs.keys()])
    content_loss = tf.reduce_mean((content_outputs['block5_conv2'] - model.outputs['block5_conv2']) ** 2)
    
    return style_weight * style_loss + content_weight * content_loss

# Run the style transfer algorithm
def neural_style_transfer(content_image, style_image, num_iterations=1000, style_weight=1e-2, content_weight=1e4):
    model, style_layers, content_layers = get_model()
    
    content_outputs = model(content_image)['block5_conv2']
    style_outputs = {name: model(style_image)[name] for name in style_layers}
    
    generated_image = tf.Variable(content_image)
    
    optimizer = tf.optimizers.Adam(learning_rate=0.02)
    
    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            model_outputs = model(generated_image)
            loss = compute_loss(model, style_outputs, content_outputs, style_weight, content_weight)
        
        grads = tape.gradient(loss, generated_image)
        optimizer.apply_gradients([(grads, generated_image)])
        
        if i % 100 == 0:
            st.write(f"Iteration {i}: loss = {loss}")
    
    return generated_image

# Streamlit interface
st.title("Neural Style Transfer")

st.sidebar.header("Settings")
num_iterations = st.sidebar.slider("Iterations", min_value=500, max_value=2000, step=100, value=1000)
style_weight = st.sidebar.slider("Style Weight", min_value=1e-5, max_value=1e-1, step=1e-5, value=1e-2)
content_weight = st.sidebar.slider("Content Weight", min_value=1e2, max_value=1e5, step=1e3, value=1e4)

content_image_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_image_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

if content_image_file and style_image_file:
    content_image = load_and_preprocess_image(content_image_file)
    style_image = load_and_preprocess_image(style_image_file)

    if st.button("Run Style Transfer"):
        generated_image = neural_style_transfer(content_image, style_image, num_iterations, style_weight, content_weight)
        st.image(generated_image[0].numpy(), caption="Generated Image", use_column_width=True)
