import os
import streamlit as st
from PIL import Image

from inference import get_result_images

human_image_names = sorted([fn[:-4] for fn in os.listdir('dataset/test_img')])

if st.sidebar.checkbox('Upload'):
    human_file = st.sidebar.file_uploader("Upload a Human Image", type=["png", "jpg", "jpeg"])
    if human_file is None:
        human_file = 'dataset/test_img/default.png'
else:
    human_image_name = st.sidebar.selectbox("Choose a Human Image", human_image_names)
    human_file = f'dataset/test_img/{human_image_name}.png'
    if not os.path.exists(human_file):
        human_file = human_file.replace('.png', '.jpg')
    st.warning("Upload a Human Image in the sidebar for Virtual-Try-On")

human = Image.open(human_file)
human.save('dataset/test_img/input.png')
st.sidebar.image(human, width=300)

result_images = get_result_images()
st.image(result_images, width=600)