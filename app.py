import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageDraw, ImageFont


def colorizer(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    prototxt = "models/colorization_deploy_v2.prototxt"
    model = "models/colorization_release_v2.caffemodel"
    points = "models/pts_in_hull.npy"

    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    return colorized

def add_text_to_image(image, text, position, font_size, color):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  
    except IOError:
        font = ImageFont.load_default()  
    draw.text(position, text, fill=color, font=font)
    return image

def apply_sticker(image, sticker, position, size, angle):
    sticker = sticker.resize((size, size)).rotate(angle)
    image.paste(sticker, position, sticker)
    return image


st.title("Vintage Vibrance")
st.sidebar.header("Choose an Action")
action = st.sidebar.selectbox("Select an action", ['Black & White to Color', 'Color to Black & White'])
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    img = np.array(original_image.convert('RGB'))

    if action == 'Black & White to Color':
        output_image = colorizer(img)
        output_image_pil = Image.fromarray(output_image)
    else:
        output_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        output_image_pil = Image.fromarray(output_image)

    st.sidebar.subheader("Adjust Image Settings")
    
    brightness = st.sidebar.slider("Brightness", 0.0, 2.0, 1.0)
    enhancer = ImageEnhance.Brightness(output_image_pil)
    img_brightness = enhancer.enhance(brightness)
    
    contrast = st.sidebar.slider("Contrast", 0.0, 2.0, 1.0)
    enhancer = ImageEnhance.Contrast(img_brightness)
    img_contrast = enhancer.enhance(contrast)
    
    blur = st.sidebar.slider("Blur", 0, 10, 0)
    if blur > 0:
        img_contrast = cv2.GaussianBlur(np.array(img_contrast), (21, 21), blur)
        img_contrast = Image.fromarray(img_contrast)
    

    rotate_angle = st.sidebar.slider("Rotate", -180, 180, 0)
    if rotate_angle != 0:
        img_contrast = img_contrast.rotate(rotate_angle)
    
    st.sidebar.subheader("Add Text to Image")
    text = st.sidebar.text_input("Enter text to add:")
    if text:
        position_x = st.sidebar.slider("Text Position X", 0, img_contrast.width, 0)
        position_y = st.sidebar.slider("Text Position Y", 0, img_contrast.height, 0)
        font_size = st.sidebar.slider("Font Size", 10, 100, 30)
        text_color = st.sidebar.color_picker("Select Text Color", "#FFFFFF") 
        img_contrast = add_text_to_image(img_contrast, text, (position_x, position_y), font_size, text_color)

    st.sidebar.subheader("Apply Sticker to Image")
    sticker_file = st.sidebar.file_uploader("Choose a sticker...", type=['png'])
    if sticker_file is not None:
        sticker_image = Image.open(sticker_file).convert("RGBA")
        sticker_x = st.sidebar.slider("Sticker X Position", 0, img_contrast.width, img_contrast.width // 2)
        sticker_y = st.sidebar.slider("Sticker Y Position", 0, img_contrast.height, img_contrast.height // 2)
        sticker_size = st.sidebar.slider("Sticker Size", 10, 300, 100)
        sticker_rotation = st.sidebar.slider("Sticker Rotation", 0, 360, 0)
        img_contrast = apply_sticker(img_contrast, sticker_image, (sticker_x, sticker_y), sticker_size, sticker_rotation)

    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(img_contrast, caption="Edited Image", use_container_width=True)

    if st.sidebar.button("Save Final Image"):
        img_contrast.save("final_image.png")
        st.success("Image saved as final_image.png")

    if st.sidebar.button("Reset"):
        st.experimental_rerun()
