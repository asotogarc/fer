# File: streamlit_app.py
import streamlit as st
from pathlib import Path
import PIL
import settings
import helper

st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #AF4D8D;
        padding: 2rem;
        border-radius: 10px;
    }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden !important;}
    </style>
    """, unsafe_allow_html=True)

model_type = st.sidebar.radio("Tipo coordenadas detección", ['YOLO'])
confidence = float(st.sidebar.slider("Formato usado: pytorch", 25, 100, 40)) / 100

if model_type == 'YOLO':
    model_path = Path(settings.DETECTION_MODEL)
#elif model_type == 'Segmentation':
#    model_path = Path(settings.SEGMENTATION_MODEL)

try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Modelo IA")
source_radio = st.sidebar.radio("Red neuronal convolucional", settings.SOURCES_LIST)

if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image", use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image", use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image', use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image', use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")
