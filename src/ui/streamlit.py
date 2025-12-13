import streamlit as st

from pipelines.yolo import run_inference


def run_streamlit_app():
    st.set_page_config(layout="wide")
    st.title(":camera: Computer vision app")

    with st.sidebar:
        st.title("Upload a picture")
        image_bytes = st.file_uploader(
            "Upload image file", type=[".png", ".jpeg"], accept_multiple_files=False
        )

        st.write("## Uploaded picture")
        if image_bytes:
            st.write("🎉 Here's what you uploaded!")
            st.image(image_bytes, width=200)
        else:
            st.warning("👈 Please upload an image first...")
            st.stop()

    st.write("## YOLOv8 Object Detection")
    res_plotted = run_inference(image_bytes)
    st.image(
        res_plotted,
        caption="Detected objects",
        width="stretch"
    )