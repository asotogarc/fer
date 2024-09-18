# File: settings.py
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
ROOT = ROOT.relative_to(Path.cwd())

IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'
RTSP = 'RTSP'
YOUTUBE = 'YouTube'
SOURCES_LIST = [WEBCAM]

IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = 'office_4.jpg'
DEFAULT_DETECT_IMAGE = 'office_4_detected.jpg'

VIDEO_DIR = ROOT / 'videos'
VIDEOS_DICT = {
    'video_1': 'video_4.mp4'
}

MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = 'best2.pt'
SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'

WEBCAM_PATH = 0

# File: helper.py
from ultralytics import YOLO
import streamlit as st
import yt_dlp
import settings
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import threading
import queue

etiquetas_es = {
    'person': 'persona',
    'bicycle': 'bicicleta',
    'car': 'coche',
    'motorcycle': 'motocicleta',
    'airplane': 'avión',
    'bus': 'autobús',
    'train': 'tren',
    'truck': 'camión',
    'boat': 'barco',
    # Añade más traducciones según sea necesario
}

def load_model(model_path):
    return YOLO(model_path)

def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = display_tracker == 'Yes'
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    image = cv2.resize(image, (720, int(720*(9/16))))
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        res = model.predict(image, conf=conf)
    res_plotted = res[0].plot()
    st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)

def get_youtube_stream_url(youtube_url):
    ydl_opts = {'format': 'best[ext=mp4]', 'no_warnings': True, 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

def play_youtube_video(conf, model):
    source_youtube = st.sidebar.text_input("YouTube Video url")
    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('Detect Objects'):
        if not source_youtube:
            st.sidebar.error("Please enter a YouTube URL")
            return

        try:
            st.sidebar.info("Extracting video stream URL...")
            stream_url = get_youtube_stream_url(source_youtube)

            st.sidebar.info("Opening video stream...")
            vid_cap = cv2.VideoCapture(stream_url)

            if not vid_cap.isOpened():
                st.sidebar.error("Failed to open video stream. Please try a different video.")
                return

            st.sidebar.success("Video stream opened successfully!")
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                else:
                    break

            vid_cap.release()

        except Exception as e:
            st.sidebar.error(f"An error occurred: {str(e)}")

def play_rtsp_stream(conf, model):
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))

class VideoProcessor:
    def __init__(self, conf, model):
        self.conf = conf
        self.model = model
        self.lock = threading.Lock()
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.processing_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.processing_thread.start()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        with self.lock:
            if self.frame_queue.full():
                self.frame_queue.get()
            self.frame_queue.put(img)
        
        try:
            result = self.result_queue.get_nowait()
        except queue.Empty:
            result = None

        if result is not None:
            return av.VideoFrame.from_ndarray(result, format="bgr24")
        else:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    def process_frames(self):
        while True:
            with self.lock:
                if self.frame_queue.empty():
                    continue
                img = self.frame_queue.get()

            results = self.model(img, conf=self.conf)
            result_plotted = results[0].plot()

            if self.result_queue.full():
                self.result_queue.get()
            self.result_queue.put(result_plotted)

def play_webcam(conf, model):
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=lambda: VideoProcessor(conf, model),
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        st.write("El streaming de la cámara web está activo. Los objetos detectados se mostrarán en tiempo real.")
    
    st.markdown("Nota: Asegúrate de permitir el acceso a la cámara cuando el navegador lo solicite.")

def play_stored_video(conf, model):
    source_vid = st.sidebar.selectbox("Choose a video...", settings.VIDEOS_DICT.keys())
    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
