from __future__ import annotations

import hashlib
import io
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import load_metadata, predict_image  # noqa: E402
from nasa_gibs import DEFAULT_EXAMPLES  # noqa: E402
from satellite_imagery import download_arcgis_world_imagery  # noqa: E402

MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.csv"
COMPARISON_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"

PRODUCT_NAME = "UrbanLens Mobility"
MOBILITY_PROFILES = {
    "baixa": {
        "signal": "Possivel espraiamento urbano",
        "focus": "Conectar periferias e reduzir viagens longas",
        "summary": (
            "A ocupacao visual dispersa pode indicar distancias maiores entre moradia, trabalho "
            "e servicos, elevando o custo do transporte e a dependencia do automovel."
        ),
        "risks": [
            "Baixa viabilidade operacional do transporte coletivo convencional.",
            "Longos deslocamentos para acessar emprego, saude e educacao.",
            "Expansao da malha viaria e maior custo de infraestrutura por habitante.",
        ],
        "actions": [
            "Mapear centralidades e destinos essenciais ainda pouco conectados.",
            "Avaliar linhas alimentadoras, transporte sob demanda e integracao tarifaria.",
            "Priorizar rotas seguras de caminhada e bicicleta para o acesso local.",
            "Coordenar novas ocupacoes com corredores de transporte coletivo existentes.",
        ],
        "validation_data": [
            "Tempo medio de viagem e matriz origem-destino.",
            "Cobertura e frequencia do transporte coletivo.",
            "Distancia a empregos, escolas, saude e comercio.",
            "Evolucao da mancha urbana e uso do solo.",
        ],
    },
    "media": {
        "signal": "Potencial de densidade equilibrada",
        "focus": "Consolidar centralidades conectadas",
        "summary": (
            "A ocupacao visual intermediaria pode favorecer uma rede multimodal eficiente, "
            "desde que haja diversidade de usos, transporte coletivo e acesso seguro a pe."
        ),
        "risks": [
            "Perda do equilibrio caso a expansao ocorra sem conexao com transporte.",
            "Dependencia gradual do automovel em bairros monofuncionais.",
            "Desigualdade de acesso entre setores do mesmo territorio.",
        ],
        "actions": [
            "Preservar a conectividade entre bairros, centralidades e corredores de transporte.",
            "Fortalecer linhas estruturais e integracao com caminhada e bicicleta.",
            "Estimular uso misto e servicos de proximidade nos eixos consolidados.",
            "Monitorar crescimento para evitar espraiamento ou saturacao futura.",
        ],
        "validation_data": [
            "Distribuicao de empregos e servicos por bairro.",
            "Acessibilidade a pe aos pontos de transporte.",
            "Frequencia, lotacao e regularidade das linhas.",
            "Mudancas recentes no uso e ocupacao do solo.",
        ],
    },
    "alta": {
        "signal": "Possivel pressao sobre a rede de mobilidade",
        "focus": "Ampliar capacidade e qualidade do acesso",
        "summary": (
            "A ocupacao visual concentrada pode sustentar transporte coletivo de alta capacidade, "
            "mas exige infraestrutura compativel para evitar saturacao, inseguranca e exclusao."
        ),
        "risks": [
            "Lotacao e baixa confiabilidade do transporte coletivo.",
            "Conflitos entre pedestres, bicicletas, onibus, cargas e automoveis.",
            "Barreiras de acesso a servicos em areas muito concentradas.",
        ],
        "actions": [
            "Medir lotacao e ampliar capacidade nos corredores de maior demanda.",
            "Qualificar calcadas, travessias, iluminacao e acesso aos pontos de transporte.",
            "Integrar bicicleta, linhas alimentadoras e transporte de alta capacidade.",
            "Reorganizar estacionamento, carga e descarga e circulacao viaria.",
        ],
        "validation_data": [
            "Lotacao por faixa horaria e velocidade operacional.",
            "Fluxo de pedestres e seguranca viaria.",
            "Cobertura de calcadas, ciclovias e areas de integracao.",
            "Capacidade de escolas, saude e demais servicos locais.",
        ],
    },
}

st.set_page_config(page_title=PRODUCT_NAME, layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f4f8f7 0%, #ffffff 26%);
    }
    .product-hero {
        padding: 2rem 2.2rem;
        border-radius: 18px;
        color: white;
        background: linear-gradient(120deg, #073b4c 0%, #087f5b 58%, #1b9aaa 100%);
        box-shadow: 0 16px 40px rgba(7, 59, 76, 0.18);
        margin-bottom: 1.2rem;
    }
    .product-hero h1 {
        color: white;
        margin: 0 0 0.35rem 0;
        font-size: 2.35rem;
    }
    .product-hero p {
        margin: 0;
        max-width: 850px;
        font-size: 1.05rem;
        opacity: 0.95;
    }
    .eyebrow {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        opacity: 0.82;
        margin-bottom: 0.5rem;
    }
    .decision-card {
        padding: 1rem 1.1rem;
        border: 1px solid #d7e5e0;
        border-radius: 12px;
        background: #ffffff;
        min-height: 132px;
    }
    .decision-card strong {
        color: #075e54;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d7e5e0;
        padding: 0.75rem 1rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_manifest():
    return pd.read_csv(MANIFEST_PATH)


@st.cache_data
def load_comparison():
    return pd.read_csv(COMPARISON_PATH)


def resolve_project_path(relative_path: str) -> Path:
    normalized_path = str(relative_path).replace("\\", "/")
    return PROJECT_ROOT.joinpath(*normalized_path.split("/"))


def build_mobility_diagnosis(prediction: dict, transport_supply: str) -> dict:
    probabilities = prediction["probabilities"]
    predicted_class = prediction["predicted_class"]
    confidence = max(probabilities.values())
    profile = MOBILITY_PROFILES[predicted_class]

    if confidence < 0.55:
        priority = "Validar evidencias"
    elif transport_supply == "Baixa" and predicted_class in {"baixa", "alta"}:
        priority = "Alta"
    elif predicted_class in {"baixa", "alta"}:
        priority = "Media-alta"
    else:
        priority = "Moderada"

    contextual_note = {
        "Baixa": "A oferta informada reforca a necessidade de investigar lacunas de acesso.",
        "Media": "A oferta informada deve ser comparada com demanda, frequencia e cobertura.",
        "Alta": "A oferta informada e positiva, mas capacidade e acesso de ultima milha ainda precisam ser medidos.",
        "Nao informada": "Informe ou cruze a oferta de transporte para refinar a prioridade.",
    }[transport_supply]

    return {
        **profile,
        "priority": priority,
        "confidence": confidence,
        "contextual_note": contextual_note,
    }


def record_analysis(
    prediction: dict,
    diagnosis: dict,
    source: str,
    reference: str,
    details: dict | None = None,
) -> None:
    probabilities = prediction["probabilities"]
    record = {
        "data_hora_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "origem": source,
        "referencia": reference,
        "classe_prevista": prediction["predicted_class"],
        "confianca": max(probabilities.values()),
        "prob_baixa": probabilities.get("baixa", 0.0),
        "prob_media": probabilities.get("media", 0.0),
        "prob_alta": probabilities.get("alta", 0.0),
        "modelo": prediction["model_name"],
        "latitude": "",
        "longitude": "",
        "data_imagem": "",
        "area_metros": "",
        "sinal_mobilidade": diagnosis["signal"],
        "prioridade": diagnosis["priority"],
        "foco": diagnosis["focus"],
        "acoes_recomendadas": " | ".join(diagnosis["actions"]),
        "dados_para_validar": " | ".join(diagnosis["validation_data"]),
        "oferta_transporte": "Nao informada",
    }
    record.update(details or {})
    st.session_state.analysis_history.append(record)


def build_analysis_report(history: list[dict]) -> bytes:
    def format_percentage(value) -> str:
        try:
            return f"{float(value):.2%}"
        except (TypeError, ValueError):
            return ""

    report_frame = pd.DataFrame(history)
    expected_columns = [
        "data_hora_utc",
        "origem",
        "referencia",
        "classe_prevista",
        "confianca",
        "sinal_mobilidade",
        "prioridade",
        "foco",
        "oferta_transporte",
        "acoes_recomendadas",
        "dados_para_validar",
        "prob_baixa",
        "prob_media",
        "prob_alta",
        "modelo",
        "latitude",
        "longitude",
        "data_imagem",
        "area_metros",
    ]
    for column in expected_columns:
        if column not in report_frame:
            report_frame[column] = ""
    report_frame = report_frame[expected_columns]
    report_frame["confianca"] = report_frame["confianca"].map(format_percentage)
    for column in ("prob_baixa", "prob_media", "prob_alta"):
        report_frame[column] = report_frame[column].map(format_percentage)

    display_names = {
        "data_hora_utc": "Data/hora (UTC)",
        "origem": "Origem",
        "referencia": "Referencia",
        "classe_prevista": "Classe prevista",
        "confianca": "Confianca",
        "sinal_mobilidade": "Sinal territorial",
        "prioridade": "Prioridade",
        "foco": "Foco recomendado",
        "oferta_transporte": "Oferta de transporte informada",
        "acoes_recomendadas": "Acoes recomendadas",
        "dados_para_validar": "Dados para validar",
        "prob_baixa": "Prob. baixa",
        "prob_media": "Prob. media",
        "prob_alta": "Prob. alta",
        "modelo": "Modelo",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "data_imagem": "Data da imagem",
        "area_metros": "Area (m)",
    }
    report_frame = report_frame.rename(columns=display_names)
    class_summary = (
        pd.DataFrame(history)["classe_prevista"]
        .value_counts()
        .rename_axis("Classe")
        .reset_index(name="Quantidade")
    )
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <title>Relatorio de analises de densidade urbana</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #172033; }}
    h1, h2 {{ color: #102a43; }}
    .summary {{ display: flex; gap: 16px; margin: 20px 0; }}
    .card {{ background: #eef4f8; border-radius: 8px; padding: 14px 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
    th, td {{ border: 1px solid #bcccdc; padding: 8px; text-align: left; }}
    th {{ background: #d9e2ec; }}
    tr:nth-child(even) {{ background: #f7f9fb; }}
    .footer {{ color: #627d98; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>UrbanLens Mobility</h1>
  <p>Relatorio executivo de triagem territorial para planejamento de mobilidade urbana.</p>
  <div class="summary">
    <div class="card"><strong>Total de analises</strong><br>{len(history)}</div>
    <div class="card"><strong>Gerado em</strong><br>{generated_at}</div>
  </div>
  <h2>Resumo por classe</h2>
  {class_summary.to_html(index=False, border=0, escape=True)}
  <h2>Diagnosticos e planos de acao</h2>
  {report_frame.to_html(index=False, border=0, escape=True)}
  <h2>Como interpretar</h2>
  <p>
    A classificacao representa densidade urbana visual percebida na imagem. Ela nao mede
    populacao, renda, demanda de viagens ou capacidade real da infraestrutura. As recomendacoes
    devem ser validadas com dados locais antes de orientar investimento publico.
  </p>
  <p class="footer">Relatorio gerado pelo {PRODUCT_NAME}.</p>
</body>
</html>
"""
    return html.encode("utf-8")


def run_prediction(
    image: Image.Image,
    analysis_key: str,
    source: str,
    reference: str,
    details: dict | None = None,
    transport_supply: str = "Nao informada",
    prediction_override=None,
) -> None:
    st.image(image, caption="Imagem analisada", use_column_width=True)
    prediction_cache = st.session_state.prediction_cache
    if analysis_key in prediction_cache:
        prediction = prediction_cache[analysis_key]
    else:
        prediction = prediction_override or predict_image(image)
        prediction_cache[analysis_key] = prediction
        diagnosis = build_mobility_diagnosis(prediction, transport_supply)
        record_analysis(prediction, diagnosis, source, reference, details)

    diagnosis = build_mobility_diagnosis(prediction, transport_supply)
    st.markdown("### Diagnostico territorial")
    metric_density, metric_confidence, metric_priority = st.columns(3)
    metric_density.metric("Padrao visual", prediction["predicted_class"].title())
    metric_confidence.metric("Confianca do modelo", f"{diagnosis['confidence']:.1%}")
    metric_priority.metric("Prioridade de investigacao", diagnosis["priority"])

    st.success(f"{diagnosis['signal']}: {diagnosis['focus']}")
    st.write(diagnosis["summary"])
    st.caption(diagnosis["contextual_note"])

    risk_column, action_column, data_column = st.columns(3)
    with risk_column:
        st.markdown("#### Riscos a investigar")
        for risk in diagnosis["risks"]:
            st.markdown(f"- {risk}")
    with action_column:
        st.markdown("#### Plano de acao inicial")
        for action in diagnosis["actions"]:
            st.markdown(f"- {action}")
    with data_column:
        st.markdown("#### Dados para validar")
        for validation_item in diagnosis["validation_data"]:
            st.markdown(f"- {validation_item}")

    with st.expander("Ver probabilidades do modelo"):
        st.bar_chart(pd.DataFrame([prediction["probabilities"]]).T.rename(columns={0: "probabilidade"}))

    st.warning(
        "Triagem de apoio a decisao: densidade visual nao equivale a densidade populacional "
        "e nao substitui dados de transporte, uso do solo ou pesquisa de campo."
    )


st.markdown(
    """
    <div class="product-hero">
      <div class="eyebrow">Inteligencia territorial para cidades</div>
      <h1>UrbanLens Mobility</h1>
      <p>
        Transforme imagens de satelite em sinais de prioridade para mobilidade urbana.
        Identifique possivel espraiamento, equilibrio ou pressao territorial e receba um
        plano inicial de investigacao para orientar decisoes publicas.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "prediction_cache" not in st.session_state:
    st.session_state.prediction_cache = {}

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

value_one, value_two, value_three = st.columns(3)
with value_one:
    st.markdown(
        '<div class="decision-card"><strong>Problema</strong><br>'
        "Crescimento disperso ou concentrado sem infraestrutura compativel.</div>",
        unsafe_allow_html=True,
    )
with value_two:
    st.markdown(
        '<div class="decision-card"><strong>Decisao apoiada</strong><br>'
        "Onde aprofundar estudos e quais lacunas de mobilidade investigar primeiro.</div>",
        unsafe_allow_html=True,
    )
with value_three:
    st.markdown(
        '<div class="decision-card"><strong>Entrega</strong><br>'
        "Diagnostico territorial, plano de acao inicial e relatorio executivo.</div>",
        unsafe_allow_html=True,
    )

diagnosis_tab, history_tab, methodology_tab = st.tabs(
    ["Diagnostico territorial", "Historico e relatorio", "Metodologia e transparencia"]
)

with diagnosis_tab:
    st.markdown("## Analise um territorio")
    st.write(
        "Escolha uma imagem propria ou busque um recorte real. O resultado indica um sinal "
        "territorial e organiza as proximas perguntas de mobilidade."
    )

    input_column, guidance_column = st.columns([1.2, 0.8])
    analysis_payload = None

    with input_column:
        transport_supply = st.selectbox(
            "Oferta atual de transporte coletivo no territorio",
            options=["Nao informada", "Baixa", "Media", "Alta"],
            help="Informacao contextual declarada pelo usuario para priorizar a investigacao.",
        )
        source_mode = st.radio(
            "Fonte da imagem",
            options=["Enviar imagem", "Buscar territorio"],
            horizontal=True,
        )

        if source_mode == "Enviar imagem":
            uploaded_file = st.file_uploader(
                "Imagem aerea ou de satelite em PNG/JPG",
                type=["png", "jpg", "jpeg"],
            )
            if uploaded_file:
                uploaded_bytes = uploaded_file.getvalue()
                upload_hash = hashlib.sha256(uploaded_bytes).hexdigest()
                analysis_payload = {
                    "image": Image.open(io.BytesIO(uploaded_bytes)),
                    "analysis_key": f"upload:{upload_hash}:{transport_supply}",
                    "source": "Upload",
                    "reference": uploaded_file.name,
                    "details": {"oferta_transporte": transport_supply},
                }
        else:
            preset_options = {example.name: example for example in DEFAULT_EXAMPLES}
            selected_preset = st.selectbox(
                "Territorio de referencia",
                options=["manual", *preset_options.keys()],
                format_func=lambda value: (
                    "Informar coordenadas"
                    if value == "manual"
                    else value.replace("_", " ").title()
                ),
            )

            if selected_preset == "manual":
                default_latitude = -23.5505
                default_longitude = -46.6333
                default_bbox_size = 12000
                location_note = "Informe o centro e a extensao aproximada do territorio."
            else:
                preset = preset_options[selected_preset]
                default_latitude = preset.latitude
                default_longitude = preset.longitude
                default_bbox_size = preset.bbox_size_meters
                location_note = preset.note

            st.caption(location_note)
            with st.form("territory_fetch_form"):
                latitude = st.number_input("Latitude", value=float(default_latitude), format="%.6f")
                longitude = st.number_input("Longitude", value=float(default_longitude), format="%.6f")
                bbox_size_meters = st.slider(
                    "Extensao analisada ao redor do ponto (metros)",
                    min_value=2000,
                    max_value=30000,
                    step=1000,
                    value=int(default_bbox_size),
                )
                submitted = st.form_submit_button(
                    "Analisar territorio",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:
                request_name = (
                    selected_preset
                    if selected_preset != "manual"
                    else f"manual_{latitude:.4f}_{longitude:.4f}"
                )
                try:
                    image_bytes, source_url, _ = download_arcgis_world_imagery(
                        latitude=latitude,
                        longitude=longitude,
                        bbox_size_meters=bbox_size_meters,
                        output_path=None,
                        image_size=1024,
                    )
                    image_hash = hashlib.sha256(image_bytes).hexdigest()
                    st.caption(f"Imagem obtida de ArcGIS World Imagery. [Ver requisicao]({source_url})")
                    analysis_payload = {
                        "image": Image.open(io.BytesIO(image_bytes)),
                        "analysis_key": f"satellite:{image_hash}:{transport_supply}",
                        "source": "ArcGIS World Imagery",
                        "reference": request_name,
                        "details": {
                            "latitude": latitude,
                            "longitude": longitude,
                            "data_imagem": "Mosaico atual do provedor",
                            "area_metros": bbox_size_meters,
                            "oferta_transporte": transport_supply,
                        },
                    }
                except Exception as error:
                    st.error(f"Nao foi possivel obter a imagem do territorio: {error}")

    with guidance_column:
        st.markdown("### O que o produto responde")
        st.markdown(
            """
            1. O padrao construido parece disperso, equilibrado ou concentrado?
            2. Qual problema de mobilidade merece investigacao primeiro?
            3. Quais acoes podem compor um plano inicial?
            4. Quais dados precisam confirmar a hipotese?
            """
        )
        st.info(
            "Foco recomendado: mobilidade urbana. Para diagnosticar acesso a moradia seriam "
            "necessarios dados adicionais de renda, preco, vacancia, deficit e zoneamento."
        )
        st.markdown("### Publico do produto")
        st.write(
            "Prefeituras, equipes de planejamento, mobilidade, desenvolvimento urbano "
            "e consultorias territoriais."
        )

    if analysis_payload:
        st.divider()
        run_prediction(
            **analysis_payload,
            transport_supply=transport_supply,
        )
    else:
        st.info("Selecione uma imagem para gerar o diagnostico e liberar o relatorio.")

with history_tab:
    st.markdown("## Carteira de territorios analisados")
    st.write(
        "Compare os sinais encontrados durante esta sessao e exporte um documento para "
        "discussao tecnica ou apresentacao."
    )
    if st.session_state.analysis_history:
        history_frame = pd.DataFrame(st.session_state.analysis_history)
        preview_columns = [
            "data_hora_utc",
            "referencia",
            "classe_prevista",
            "confianca",
            "sinal_mobilidade",
            "prioridade",
        ]
        for column in preview_columns:
            if column not in history_frame:
                history_frame[column] = ""
        history_preview = history_frame[preview_columns].copy()
        history_preview["confianca"] = history_preview["confianca"].map(
            lambda value: f"{value:.2%}"
        )
        history_preview = history_preview.rename(
            columns={
                "data_hora_utc": "Data/hora UTC",
                "referencia": "Territorio",
                "classe_prevista": "Padrao visual",
                "confianca": "Confianca",
                "sinal_mobilidade": "Sinal territorial",
                "prioridade": "Prioridade",
            }
        )
        st.dataframe(history_preview.iloc[::-1], use_container_width=True, hide_index=True)

        export_column, clear_column = st.columns([3, 1])
        with export_column:
            st.download_button(
                "Exportar relatorio executivo",
                data=build_analysis_report(st.session_state.analysis_history),
                file_name=f"urbanlens_mobility_{date.today().isoformat()}.html",
                mime="text/html",
                type="primary",
                use_container_width=True,
            )
        with clear_column:
            if st.button("Limpar historico", use_container_width=True):
                st.session_state.analysis_history = []
                st.session_state.prediction_cache = {}
                st.rerun()
    else:
        st.info("As analises feitas nesta sessao aparecerao aqui.")
        st.download_button(
            "Exportar relatorio executivo",
            data=b"",
            file_name=f"urbanlens_mobility_{date.today().isoformat()}.html",
            mime="text/html",
            disabled=True,
            use_container_width=True,
        )
        st.caption("Analise ao menos um territorio para liberar o relatorio.")

with methodology_tab:
    st.markdown("## Metodologia e limites")
    st.warning(
        "O modelo classifica densidade urbana visual. Ele nao estima diretamente populacao, "
        "renda, demanda de viagens, congestionamento ou qualidade da infraestrutura."
    )

    model_column, validation_column, use_column = st.columns(3)
    model_column.metric("Modelo em producao", best_model_name)
    validation_column.metric(
        "Melhor acuracia de validacao",
        f"{metadata['best_validation_accuracy'] * 100:.2f}%",
    )
    use_column.metric("Acuracia em teste", f"{metadata['test_accuracy'] * 100:.2f}%")

    with st.expander("Desempenho e comparacao dos modelos"):
        st.dataframe(comparison, use_container_width=True)
        visual_left, visual_right = st.columns(2)
        with visual_left:
            st.markdown("#### Matriz de confusao")
            if confusion_matrix_path.exists():
                st.image(str(confusion_matrix_path), use_column_width=True)
        with visual_right:
            st.markdown("#### Exemplos de erros")
            if error_examples_path.exists():
                st.image(str(error_examples_path), use_column_width=True)

    with st.expander("Amostras usadas no projeto"):
        sample_rows = manifest.groupby("visual_density_class").head(2).reset_index(drop=True)
        for start_index in range(0, len(sample_rows), 3):
            sample_columns = st.columns(3)
            for column, (_, row) in zip(
                sample_columns,
                sample_rows.iloc[start_index:start_index + 3].iterrows(),
            ):
                image_path = resolve_project_path(row["local_image_path"])
                caption = f"{row['visual_density_class']} | {row['zone_type']}"
                if image_path.is_file():
                    column.image(str(image_path), caption=caption, use_column_width=True)
                else:
                    column.warning(f"Imagem indisponivel: {caption}")

    st.markdown("### Referencias de planejamento")
    st.markdown(
        """
        - [WRI Brasil - Desenvolvimento Orientado ao Transporte Sustentavel](https://www.wribrasil.org.br/projetos/desenvolvimento-orientado-ao-transporte-sustentavel-dots)
        - [ITDP - Principios de Desenvolvimento Orientado ao Transporte](https://itdp.org/publication/principios-desenvolvimento-orientado-ao-transporte/)
        - [UN-Habitat - Planejamento e desenho urbano](https://unhabitat.org/planning-and-design/)
        """
    )
    st.caption(f"Artefato do modelo em uso: {best_artifact_path}")
