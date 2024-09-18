import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO

# Etiquetas en español (puedes expandir este diccionario según sea necesario)
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
    """
    Carga un modelo de detección de objetos YOLO desde el model_path especificado.
    
    Parámetros:
        model_path (str): La ruta al archivo del modelo YOLO.
    Retorna:
        Un modelo de detección de objetos YOLO.
    """
    model = YOLO(model_path)
    return model

class VideoProcessor:
    def __init__(self, model, conf):
        self.model = model
        self.conf = conf

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Realizar detección de objetos
        results = self.model(img, conf=self.conf)
        
        # Dibujar los objetos detectados en la imagen
        annotated_frame = results[0].plot()
        
        # Traducir etiquetas al español
        for det in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = det
            class_name = results[0].names[int(class_id)]
            translated_name = etiquetas_es.get(class_name, class_name)  # Usar original si no hay traducción
            label = f"{translated_name} {score:.2f}"
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

def main():
    st.title("Detección de Objetos en Tiempo Real con YOLO (Modelo best2.pt)")

    # Barra lateral para el umbral de confianza
    st.sidebar.header("Configuración")
    confidence_threshold = st.sidebar.slider("Umbral de Confianza", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    # Cargar el modelo YOLO best2.pt
    model_path = "best2.pt"
    model = load_model(model_path)

    # Crear una instancia de VideoProcessor
    processor = VideoProcessor(model, confidence_threshold)

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=lambda: processor,
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        st.write("El streaming de la cámara web está activo. Los objetos detectados se mostrarán en tiempo real.")
    
    st.markdown("Nota: Asegúrate de permitir el acceso a la cámara cuando el navegador lo solicite.")

if __name__ == "__main__":
    main()
