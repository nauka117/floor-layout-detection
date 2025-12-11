
import streamlit as st
from ultralytics import YOLO
from PIL import Image

def main():
    st.set_page_config(layout="wide")
    st.title(":camera: Computer vision app")


    # Let user upload a picture
    with st.sidebar:
        st.title("Upload a picture")

        
        image_bytes = st.file_uploader(
            "Upload image file", type=[".png"], accept_multiple_files=False
        )

    st.write("## Uploaded picture")
    if image_bytes:
        st.write("🎉 Here's what you uploaded!")
        st.image(image_bytes, width=200)
    else:
        st.warning("👈 Please upload an image first...")
        st.stop()


    st.write("## YOLOv8 Object Detection")

    yolov8_model = YOLO("src/models/yolo-v8/pretrained.pt")

    image = Image.open(image_bytes)

    result = yolov8_model.predict(image)

    res_plotted = result[0].plot()[:,:, ::-1]

    st.image(res_plotted, caption="Detected objects", use_column_width=True)
    

if __name__ == "__main__":
    main()