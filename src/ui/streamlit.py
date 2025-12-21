import sys
import json
from pathlib import Path


src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import streamlit as st

from pipelines.yolo import detect_layout, get_detection_results
from utils.visualization import create_annotated_image
from utils.json_export import format_walls_to_json


def run_streamlit_app():
    st.set_page_config(layout="wide", page_title="Floor Layout Detection")
    st.title("🏠 Floor Layout Detection")
    st.markdown("Загрузите изображение плана квартиры для извлечения геометрии стен")

    with st.sidebar:
        st.title("📤 Загрузка изображения")
        image_bytes = st.file_uploader(
            "Выберите файл изображения",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False
        )

        if image_bytes:
            st.write("✅ Изображение загружено")
            st.image(image_bytes, width=200)
        else:
            st.warning("👈 Пожалуйста, загрузите изображение...")
            st.stop()


    with st.spinner("Обработка изображения..."):
        walls, source_name = detect_layout(image_bytes)
        
        result_json = format_walls_to_json(walls, source_name)
        
        result, walls, image = get_detection_results(image_bytes)
        annotated_image = create_annotated_image(image_bytes, result, walls)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Результаты детекции")
        st.json(result_json)
        
        json_str = json.dumps(result_json, indent=2, ensure_ascii=False)
        st.download_button(
            label="💾 Скачать JSON",
            data=json_str,
            file_name=f"{result_json['meta']['source']}_result.json",
            mime="application/json"
        )
    
    with col2:
        st.subheader("🖼️ Визуализация")
        st.image(
            annotated_image,
            caption=f"Найдено стен: {len(result_json['walls'])}",
            use_container_width=True
        )
        
        st.info(f"**Статистика:**\n- Найдено стен: {len(result_json['walls'])}")


if __name__ == "__main__":
    run_streamlit_app()