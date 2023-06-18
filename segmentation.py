import os
import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from streamlit_image_comparison import image_comparison

model = YOLO('yolov8n.pt')

st.title("Image Segmentation with YOLOv8: A Web Integration")
st.subheader("Implementing for Image Segmentation and Object Detection")
st.write("Image segmentation is a critical task in computer vision that involves dividing an image into multiple "
         "segments or regions. YOLOv8 is a state-of-the-art deep learning model that can be used for "
         "image segmentation and object detection. In this web app, we will implement an interesting example "
         "using YOLOv8 for image segmentation. We can simply drop an image, View the identified segment "
         "a piece, a whole and the distribution of the identified segments. The essence of this application "
         "is to build a practical understanding and implementation of the powerful and light YOLOv8 "
         "for image segmentation and object detection")

url = "https://www.example.com"
link = f'<a href="{url}">Click here</a>'
st.markdown(link, unsafe_allow_html=True)

st.divider()

st.markdown('')
st.markdown('##### Segmented Pieces')

img_file = 'bus.jpg'
# uploaded_file = st.sidebar.file_uploader("Upload your Image here", type=['png', 'jpeg', 'jpg'])
uploaded_file = st.sidebar.file_uploader("Drop a JPG file", accept_multiple_files=False, type='jpg')
if uploaded_file is not None:
    # uploaded_file.name = "uploaded_image"
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}

    parent_media_path = "media-directory"
    new_file_name = "uploaded_image.jpg"
    with open(os.path.join(parent_media_path, new_file_name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    img_file = os.path.join(parent_media_path, new_file_name)

    st.sidebar.success("File saved successfully")
    print(f"File saved successfully to {os.path.abspath(os.path.join(parent_media_path, new_file_name))}")
else:
    st.sidebar.write("You are using a placeholder image, Upload your Image (.jpg for now) to explore")


results = model(img_file)
img = cv2.imread(img_file)
names_list = []
for result in results:
    boxes = result.boxes.cpu().numpy()
    numCols = len(boxes)
    cols = st.columns(numCols)
    for box in boxes:
        r = box.xyxy[0].astype(int)
        rect = cv2.rectangle(img, r[:2], r[2:], (255, 55, 255), 2)
    # st.image(rect)
    # render image-comparison

    st.markdown('')
    st.markdown('##### Slider of Uploaded Image and Segments')
    image_comparison(
        img1=img_file,
        img2=img,
        label1="Actual Image",
        label2="Segmented Image",
        width=700,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True
    )
    for i, box in enumerate(boxes):
        r = box.xyxy[0].astype(int)
        crop = img[r[1]:r[3], r[0]:r[2]]
        predicted_name = result.names[int(box.cls[0])]
        names_list.append(predicted_name)
        with cols[i]:
            st.write(str(predicted_name) + ".jpg")
            st.image(crop)

st.sidebar.divider()
st.sidebar.markdown('')
st.sidebar.markdown('#### Distribution of identified items')

df_x = pd.DataFrame(names_list)
summary_table = df_x[0].value_counts().rename_axis('unique_values').reset_index(name='counts').T
st.sidebar.dataframe(summary_table)



st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.sidebar.divider()
st.sidebar.info("Made with ❤ by the AfroLogicInsect")