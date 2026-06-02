from __future__ import annotations

import io
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import load_metadata, predict_image  # noqa: E402
from nasa_gibs import (  # noqa: E402
    DEFAULT_DATE,
    DEFAULT_EXAMPLES,
    build_real_image_output_path,
)
from satellite_imagery import download_arcgis_world_imagery  # noqa: E402

MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.csv"
COMPARISON_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"

st.set_page_config(page_title="ACV - Densidade Urbana", layout="wide")


@st.cache_data
def load_manifest():
    return pd.read_csv(MANIFEST_PATH)


@st.cache_data
def load_comparison():
    return pd.read_csv(COMPARISON_PATH)


def run_prediction(image: Image.Image, prediction_override=None) -> None:
    st.image(image, caption="Imagem analisada", width="stretch")
    prediction = prediction_override or predict_image(image)
    st.success(
        f"Classe prevista: {prediction['predicted_class'].upper()} | Modelo: {prediction['model_name']}"
    )
    st.bar_chart(pd.DataFrame([prediction["probabilities"]]).T.rename(columns={0: "probabilidade"}))


st.title("Applied Computer Vision - Densidade Urbana por Imagem de Satelite")
st.caption(
    "Classificacao de densidade urbana visual com visao computacional, incluindo teste com imagens reais de satelite."
)

if not MANIFEST_PATH.exists() or not COMPARISON_PATH.exists():
    st.error("Artefatos nao encontrados. Rode primeiro `python src/prepare_dataset.py` e `python src/train.py`.")
    st.stop()

metadata = load_metadata()
comparison = load_comparison()
manifest = load_manifest()
best_model_name = metadata["best_model"]
confusion_matrix_path = PROJECT_ROOT / "reports" / f"confusion_matrix_{best_model_name}.png"
error_examples_path = PROJECT_ROOT / "reports" / f"error_examples_{best_model_name}.png"
best_artifact_path = (
    "models/urban_meta_stack_v1.joblib"
    if best_model_name == "urban_meta_stack_v1"
    else "models/best_model.pt"
)

left_column, right_column = st.columns([1.0, 1.0])

with left_column:
    st.subheader("Teste com nova imagem")
    uploaded_file = st.file_uploader("Envie uma imagem PNG/JPG", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        run_prediction(Image.open(uploaded_file))
    else:
        st.info("Envie uma imagem ou use a busca de satelite logo abaixo para rodar a inferencia do melhor modelo.")

    st.subheader("Baixar imagem real de satelite")
    preset_options = {example.name: example for example in DEFAULT_EXAMPLES}
    selected_preset = st.selectbox(
        "Preset de local real",
        options=["manual", *preset_options.keys()],
        format_func=lambda value: "Escolher manualmente" if value == "manual" else value.replace("_", " ").title(),
    )

    default_date = date.fromisoformat(DEFAULT_DATE)
    if selected_preset == "manual":
        default_latitude = -23.5505
        default_longitude = -46.6333
        default_bbox_size = 12000
        location_note = "Informe coordenadas reais para buscar um recorte nitido de satelite."
    else:
        preset = preset_options[selected_preset]
        default_latitude = preset.latitude
        default_longitude = preset.longitude
        default_bbox_size = preset.bbox_size_meters
        location_note = preset.note

    st.caption(location_note)
    with st.form("nasa_fetch_form"):
        latitude = st.number_input("Latitude", value=float(default_latitude), format="%.6f")
        longitude = st.number_input("Longitude", value=float(default_longitude), format="%.6f")
        bbox_size_meters = st.slider(
            "Area capturada ao redor do ponto (metros)",
            min_value=2000,
            max_value=30000,
            step=1000,
            value=int(default_bbox_size),
        )
        image_date = st.date_input("Data da imagem", value=default_date)
        submitted = st.form_submit_button("Baixar imagem e classificar")

    if submitted:
        request_name = selected_preset if selected_preset != "manual" else f"manual_{latitude:.4f}_{longitude:.4f}"
        output_path = build_real_image_output_path(request_name, image_date.isoformat())
        try:
            image_bytes, source_url, saved_path = download_arcgis_world_imagery(
                latitude=latitude,
                longitude=longitude,
                bbox_size_meters=bbox_size_meters,
                output_path=output_path,
                image_size=1024,
            )
            fetched_image = Image.open(io.BytesIO(image_bytes))
            st.caption(f"Fonte ArcGIS World Imagery: {source_url}")
            if saved_path is not None:
                st.caption(f"Imagem salva em: {saved_path}")
            run_prediction(fetched_image)
        except Exception as error:
            st.error(f"Nao foi possivel baixar a imagem real de satelite: {error}")

with right_column:
    st.subheader("Comparacao entre arquiteturas")
    st.dataframe(comparison, use_container_width=True)
    st.metric("Melhor modelo", best_model_name)
    st.metric("Melhor acuracia de validacao", f"{metadata['best_validation_accuracy'] * 100:.2f}%")
    st.metric("Acuracia em teste", f"{metadata['test_accuracy'] * 100:.2f}%")

st.subheader("Amostras do dataset")
sample_rows = manifest.groupby("visual_density_class").head(2).reset_index(drop=True)
for start_index in range(0, len(sample_rows), 3):
    sample_columns = st.columns(3)
    for column, (_, row) in zip(sample_columns, sample_rows.iloc[start_index:start_index + 3].iterrows()):
        image_path = PROJECT_ROOT / row["local_image_path"]
        column.image(
            str(image_path),
            caption=f"{row['visual_density_class']} | {row['zone_type']}",
            width="stretch",
        )

visual_left, visual_right = st.columns(2)
with visual_left:
    st.subheader("Matriz de confusao do melhor modelo")
    if confusion_matrix_path.exists():
        st.image(str(confusion_matrix_path), width="stretch")

with visual_right:
    st.subheader("Exemplos de erros")
    if error_examples_path.exists():
        st.image(str(error_examples_path), width="stretch")

with st.expander("Sobre o projeto"):
    st.markdown(
        f"""
        - Problema: classificar densidade urbana visual (`baixa`, `media`, `alta`) a partir de imagens de satelite.
        - Conexao com a Global Solution: apoiar a estimativa e o monitoramento de ocupacao urbana.
        - Tecnicas comparadas: `UrbanDensityCNNV1`, `UrbanDensityCNNV2`, `UrbanDensityCNNV3`, `UrbanDensityCNNV4` e um stacker logistico final.
        - Melhor artefato atual: `{best_artifact_path}`.
        """
    )
