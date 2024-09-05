import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Enable eager execution for TensorFlow
tf.config.run_functions_eagerly(True)

# Load and preprocess image
def load_and_preprocess_image(image_file, target_size=(512, 512)):
    try:
        img = Image.open(image_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(target_size)
        img = np.array(img)
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img[tf.newaxis, :]
        return img
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

# Deprocess image
def deprocess_image(image):
    image = image.numpy()
    image = image.squeeze()
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)

# Define the model
def get_model():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layers = ['block5_conv2']
    
    outputs = [vgg.get_layer(name).output for name in (style_layers + content_layers)]
    model = tf.keras.Model([vgg.input], outputs)
    
    return model, style_layers, content_layers

# Define the compute_loss function
def compute_loss(style_outputs, content_outputs, generated_style_outputs, generated_content_outputs, style_weight=1e-2, content_weight=1e4):
    style_loss = tf.add_n([tf.reduce_mean((generated_style_outputs[layer] - style_outputs[layer]) ** 2) for layer in style_outputs])
    style_loss *= style_weight / len(style_outputs)
    content_loss = tf.reduce_mean((generated_content_outputs['block5_conv2'] - content_outputs['block5_conv2']) ** 2)
    content_loss *= content_weight
    total_loss = style_loss + content_loss
    return total_loss

# Neural Style Transfer function
def neural_style_transfer(content_image, style_image, num_iterations=2000, style_weight=1e-2, content_weight=1e4, learning_rate=0.01):
    model, style_layers, content_layers = get_model()
    
    content_targets = model(content_image)
    content_outputs = {content_layers[0]: content_targets[len(style_layers):][0]}
    
    style_targets = model(style_image)
    style_outputs = {style_layers[i]: style_targets[i] for i in range(len(style_layers))}
    
    generated_image = tf.Variable(content_image)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    
    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            model_outputs = model(generated_image)
            generated_style_outputs = {style_layers[i]: model_outputs[i] for i in range(len(style_layers))}
            generated_content_outputs = {content_layers[0]: model_outputs[len(style_layers):][0]}
            
            loss = compute_loss(style_outputs, content_outputs, generated_style_outputs, generated_content_outputs, style_weight, content_weight)
        
        grads = tape.gradient(loss, generated_image)
        optimizer.apply_gradients([(grads, generated_image)])
        generated_image.assign(tf.clip_by_value(generated_image, 0.0, 1.0))
        
        if i % 100 == 0:
            st.write(f"Iteration {i}: loss = {loss}")

    return generated_image

# Streamlit interface
st.title("Neural Style Transfer")
st.write("Upload a content image and a style image to generate a stylized image.")

content_image_file = st.file_uploader("Upload Content Image", type=["jpg", "png"])
style_image_file = st.file_uploader("Upload Style Image", type=["jpg", "png"])

if content_image_file and style_image_file:
    content_image = load_and_preprocess_image(content_image_file)
    style_image = load_and_preprocess_image(style_image_file)

    if content_image is not None and style_image is not None:
        st.image(content_image[0].numpy(), caption="Content Image", use_column_width=True)
        st.image(style_image[0].numpy(), caption="Style Image", use_column_width=True)

        if st.button("Run Style Transfer"):
            generated_image = neural_style_transfer(content_image, style_image)
            
            # Clamp the generated image to [0, 1] before displaying
            generated_image_clamped = tf.clip_by_value(generated_image, 0.0, 1.0)
            st.image(generated_image_clamped[0].numpy(), caption="Generated Image", use_column_width=True)

            final_image = deprocess_image(generated_image_clamped[0])
            st.image(final_image, caption="Final Generated Image", use_column_width=True)
