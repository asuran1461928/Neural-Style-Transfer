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
# Updated compute_loss function
def compute_loss(style_outputs, content_outputs, generated_style_outputs, generated_content_outputs, style_weight=1e-2, content_weight=1e4):
    # Calculate the style loss
    style_loss = tf.add_n([tf.reduce_mean((generated_style_outputs[layer] - style_outputs[layer]) ** 2) for layer in style_outputs])
    style_loss *= style_weight / len(style_outputs)
    
    # Calculate the content loss
    content_loss = tf.reduce_mean((generated_content_outputs['block5_conv2'] - content_outputs['block5_conv2']) ** 2)
    content_loss *= content_weight
    
    # Total loss
    total_loss = style_loss + content_loss
    return total_loss

# Updated neural_style_transfer function
def neural_style_transfer(content_image, style_image, num_iterations=1000, style_weight=1e-2, content_weight=1e4):
    tf.keras.backend.clear_session()  # Reset the TensorFlow graph
    model, style_layers, content_layers = get_model()
    
    # Extract content and style outputs from the images
    content_targets = model(content_image)
    content_outputs = {content_layers[0]: content_targets[len(style_layers):][0]}
    
    style_targets = model(style_image)
    style_outputs = {style_layers[i]: style_targets[i] for i in range(len(style_layers))}
    
    # Initialize the generated image as a variable
    generated_image = tf.Variable(content_image)
    
    # Optimizer
    optimizer = tf.optimizers.Adam(learning_rate=0.02)
    
    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            model_outputs = model(generated_image)
            generated_style_outputs = {style_layers[i]: model_outputs[i] for i in range(len(style_layers))}
            generated_content_outputs = {content_layers[0]: model_outputs[len(style_layers):][0]}
            
            # Calculate loss using the updated compute_loss function
            loss = compute_loss(style_outputs, content_outputs, generated_style_outputs, generated_content_outputs, style_weight, content_weight)
        
        # Apply gradients
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
