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
        "signal": "Possível espraiamento urbano",
        "focus": "Conectar periferias e reduzir viagens longas",
        "summary": (
            "A ocupação visual dispersa pode indicar distâncias maiores entre moradia, trabalho "
            "e serviços, elevando o custo do transporte e a dependência do automóvel."
        ),
        "risks": [
            "Baixa viabilidade operacional do transporte coletivo convencional.",
            "Longos deslocamentos para acessar emprego, saúde e educação.",
            "Expansão da malha viária e maior custo de infraestrutura por habitante.",
        ],
        "actions": [
            "Mapear centralidades e destinos essenciais ainda pouco conectados.",
            "Avaliar linhas alimentadoras, transporte sob demanda e integração tarifária.",
            "Priorizar rotas seguras de caminhada e bicicleta para o acesso local.",
            "Coordenar novas ocupações com corredores de transporte coletivo existentes.",
        ],
        "validation_data": [
            "Tempo médio de viagem e matriz origem-destino.",
            "Cobertura e frequência do transporte coletivo.",
            "Distância a empregos, escolas, saúde e comércio.",
            "Evolução da mancha urbana e uso do solo.",
        ],
    },
    "media": {
        "signal": "Potencial de densidade equilibrada",
        "focus": "Consolidar centralidades conectadas",
        "summary": (
            "A ocupação visual intermediária pode favorecer uma rede multimodal eficiente, "
            "desde que haja diversidade de usos, transporte coletivo e acesso seguro a pé."
        ),
        "risks": [
            "Perda do equilíbrio caso a expansão ocorra sem conexão com transporte.",
            "Dependência gradual do automóvel em bairros monofuncionais.",
            "Desigualdade de acesso entre setores do mesmo território.",
        ],
        "actions": [
            "Preservar a conectividade entre bairros, centralidades e corredores de transporte.",
            "Fortalecer linhas estruturais e integração com caminhada e bicicleta.",
            "Estimular uso misto e serviços de proximidade nos eixos consolidados.",
            "Monitorar crescimento para evitar espraiamento ou saturação futura.",
        ],
        "validation_data": [
            "Distribuição de empregos e serviços por bairro.",
            "Acessibilidade a pé aos pontos de transporte.",
            "Frequência, lotação e regularidade das linhas.",
            "Mudanças recentes no uso e ocupação do solo.",
        ],
    },
    "alta": {
        "signal": "Possível pressão sobre a rede de mobilidade",
        "focus": "Ampliar capacidade e qualidade do acesso",
        "summary": (
            "A ocupação visual concentrada pode sustentar transporte coletivo de alta capacidade, "
            "mas exige infraestrutura compatível para evitar saturação, insegurança e exclusão."
        ),
        "risks": [
            "Lotação e baixa confiabilidade do transporte coletivo.",
            "Conflitos entre pedestres, bicicletas, ônibus, cargas e automóveis.",
            "Barreiras de acesso a serviços em áreas muito concentradas.",
        ],
        "actions": [
            "Medir lotação e ampliar capacidade nos corredores de maior demanda.",
            "Qualificar calçadas, travessias, iluminação e acesso aos pontos de transporte.",
            "Integrar bicicleta, linhas alimentadoras e transporte de alta capacidade.",
            "Reorganizar estacionamento, carga e descarga e circulação viária.",
        ],
        "validation_data": [
            "Lotação por faixa horária e velocidade operacional.",
            "Fluxo de pedestres e segurança viária.",
            "Cobertura de calçadas, ciclovias e áreas de integração.",
            "Capacidade de escolas, saúde e demais serviços locais.",
        ],
    },
}

st.set_page_config(page_title=PRODUCT_NAME, layout="wide")
st.markdown(
    """
    <style>
    :root {
        --brand-950: #063c3b;
        --brand-800: #075e54;
        --brand-700: #087f5b;
        --brand-500: #1b9aaa;
        --ink-900: #102a43;
        --ink-700: #334e68;
        --ink-500: #627d98;
        --line: #d9e5e2;
        --surface: #ffffff;
        --surface-soft: #f3f8f6;
    }
    .stApp {
        color: var(--ink-900);
        background: #f7faf9;
    }
    .block-container {
        max-width: 1180px;
        padding-top: 1rem;
        padding-bottom: 4rem;
    }
    header[data-testid="stHeader"] {
        display: none;
    }
    .product-nav {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.85rem;
    }
    .brand-mark {
        display: grid;
        place-items: center;
        width: 38px;
        height: 38px;
        border-radius: 11px;
        color: #ffffff;
        background: var(--brand-800);
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.04em;
    }
    .brand-copy {
        line-height: 1.05;
    }
    .brand-copy strong {
        display: block;
        font-size: 1rem;
        color: var(--ink-900);
    }
    .brand-copy span {
        font-size: 0.73rem;
        color: var(--ink-500);
    }
    .product-pill {
        margin-left: auto;
        padding: 0.36rem 0.7rem;
        border: 1px solid #b9d8ce;
        border-radius: 999px;
        color: var(--brand-800);
        background: #eaf5f1;
        font-size: 0.72rem;
        font-weight: 700;
    }
    .product-hero {
        display: grid;
        grid-template-columns: minmax(0, 1.5fr) minmax(250px, 0.65fr);
        gap: 2rem;
        align-items: center;
        padding: 1.7rem 1.9rem;
        border-radius: 20px;
        color: white;
        background: linear-gradient(120deg, #073b4c 0%, #087f5b 58%, #1b9aaa 100%);
        box-shadow: 0 14px 32px rgba(7, 59, 76, 0.14);
        margin-bottom: 0.85rem;
    }
    .product-hero h1 {
        color: white;
        margin: 0 0 0.45rem 0;
        max-width: 760px;
        font-size: clamp(1.85rem, 4vw, 2.7rem);
        line-height: 1.08;
    }
    .product-hero p {
        margin: 0;
        max-width: 720px;
        font-size: 0.98rem;
        line-height: 1.55;
        opacity: 0.92;
    }
    .eyebrow {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        opacity: 0.82;
        margin-bottom: 0.55rem;
    }
    .hero-outcome {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(255, 255, 255, 0.24);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.10);
        backdrop-filter: blur(6px);
    }
    .hero-outcome span {
        display: block;
        margin-bottom: 0.35rem;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.75;
    }
    .hero-outcome strong {
        font-size: 1rem;
        line-height: 1.4;
    }
    .impact-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        overflow: hidden;
        margin-bottom: 0.9rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: var(--surface);
    }
    .impact-item {
        padding: 0.8rem 1rem;
        border-right: 1px solid var(--line);
    }
    .impact-item:last-child {
        border-right: 0;
    }
    .impact-item strong {
        display: block;
        color: var(--brand-800);
        font-size: 1.25rem;
        line-height: 1.1;
    }
    .impact-item span {
        display: block;
        margin-top: 0.25rem;
        color: var(--ink-700);
        font-size: 0.78rem;
        line-height: 1.3;
    }
    .impact-source {
        color: var(--ink-500);
        font-size: 0.65rem;
    }
    .section-kicker {
        margin-bottom: 0.25rem;
        color: var(--brand-700);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .section-title {
        margin: 0 0 0.35rem 0;
        color: var(--ink-900);
        font-size: 1.75rem;
        line-height: 1.2;
    }
    .section-copy {
        margin: 0 0 1rem 0;
        color: var(--ink-700);
        font-size: 0.95rem;
    }
    .mini-list {
        margin: 0;
        padding-left: 1.15rem;
        color: var(--ink-700);
    }
    .mini-list li {
        margin-bottom: 0.55rem;
        line-height: 1.4;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line);
        border-radius: 16px;
        background: var(--surface);
    }
    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        padding: 0.75rem 1rem;
        border-radius: 12px;
    }
    div[data-testid="stFileUploaderDropzone"] {
        border: 1px dashed #9fc6ba;
        border-radius: 12px;
        background: var(--surface-soft);
    }
    button[kind="primary"],
    div[data-testid="stFormSubmitButton"] button,
    .stDownloadButton button[kind="primary"] {
        color: #ffffff !important;
        border-color: var(--brand-700) !important;
        background: var(--brand-700) !important;
    }
    button[kind="primary"]:hover,
    div[data-testid="stFormSubmitButton"] button:hover,
    .stDownloadButton button[kind="primary"]:hover {
        border-color: var(--brand-800) !important;
        background: var(--brand-800) !important;
    }
    div[data-baseweb="tab-list"] {
        gap: 0.25rem;
        padding: 0.25rem;
        border: 1px solid var(--line);
        border-radius: 12px;
        background: #edf4f1;
    }
    button[data-baseweb="tab"] {
        min-height: 38px;
        padding: 0.45rem 0.8rem;
        border-radius: 9px;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--brand-800);
        background: var(--surface);
        box-shadow: 0 2px 8px rgba(16, 42, 67, 0.08);
    }
    div[data-baseweb="tab-highlight"] {
        background-color: var(--brand-700);
    }
    div[data-testid="stImage"] img {
        border-radius: 14px;
    }
    @media (max-width: 720px) {
        .block-container {
            padding-top: 0.9rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .product-pill {
            display: none;
        }
        .product-hero {
            grid-template-columns: 1fr;
            gap: 0.5rem;
            padding: 1.25rem;
            border-radius: 16px;
        }
        .product-hero h1 {
            font-size: 1.8rem;
        }
        .product-hero p {
            font-size: 0.9rem;
        }
        .hero-outcome {
            display: none;
        }
        .impact-item {
            padding: 0.7rem 0.6rem;
        }
        .impact-item strong {
            font-size: 1.05rem;
        }
        .impact-item span {
            font-size: 0.67rem;
        }
        .impact-source {
            display: none;
        }
        div[data-baseweb="tab-list"] {
            overflow-x: auto;
        }
        button[data-baseweb="tab"] {
            white-space: nowrap;
            font-size: 0.78rem;
        }
        .section-title {
            font-size: 1.45rem;
        }
        .section-copy {
            margin-bottom: 0.65rem;
            font-size: 0.86rem;
            line-height: 1.45;
        }
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
        priority = "Validar evidências"
    elif transport_supply == "Baixa" and predicted_class in {"baixa", "alta"}:
        priority = "Alta"
    elif predicted_class in {"baixa", "alta"}:
        priority = "Média-alta"
    else:
        priority = "Moderada"

    contextual_note = {
        "Baixa": "A oferta informada reforça a necessidade de investigar lacunas de acesso.",
        "Media": "A oferta informada deve ser comparada com demanda, frequência e cobertura.",
        "Alta": "A oferta informada é positiva, mas capacidade e acesso de última milha ainda precisam ser medidos.",
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
        "referencia": "Referência",
        "classe_prevista": "Classe prevista",
        "confianca": "Confiança",
        "sinal_mobilidade": "Sinal territorial",
        "prioridade": "Prioridade",
        "foco": "Foco recomendado",
        "oferta_transporte": "Oferta de transporte informada",
        "acoes_recomendadas": "Ações recomendadas",
        "dados_para_validar": "Dados para validar",
        "prob_baixa": "Prob. baixa",
        "prob_media": "Prob. media",
        "prob_alta": "Prob. alta",
        "modelo": "Modelo",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "data_imagem": "Data da imagem",
        "area_metros": "Área (m)",
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
  <title>Relatório de análises de densidade urbana</title>
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
  <p>Relatório executivo de triagem territorial para planejamento de mobilidade urbana.</p>
  <div class="summary">
    <div class="card"><strong>Total de análises</strong><br>{len(history)}</div>
    <div class="card"><strong>Gerado em</strong><br>{generated_at}</div>
  </div>
  <h2>Resumo por classe</h2>
  {class_summary.to_html(index=False, border=0, escape=True)}
  <h2>Diagnósticos e planos de ação</h2>
  {report_frame.to_html(index=False, border=0, escape=True)}
  <h2>Como interpretar</h2>
  <p>
    A classificação representa densidade urbana visual percebida na imagem. Ela não mede
    população, renda, demanda de viagens ou capacidade real da infraestrutura. As recomendações
    devem ser validadas com dados locais antes de orientar investimento público.
  </p>
  <p class="footer">Relatório gerado pelo {PRODUCT_NAME}.</p>
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
    prediction_cache = st.session_state.prediction_cache
    if analysis_key in prediction_cache:
        prediction = prediction_cache[analysis_key]
    else:
        prediction = prediction_override or predict_image(image)
        prediction_cache[analysis_key] = prediction
        diagnosis = build_mobility_diagnosis(prediction, transport_supply)
        record_analysis(prediction, diagnosis, source, reference, details)

    diagnosis = build_mobility_diagnosis(prediction, transport_supply)
    st.markdown('<div class="section-kicker">Último diagnóstico</div>', unsafe_allow_html=True)

    image_column, result_column = st.columns([0.72, 1.28], gap="large")
    with image_column:
        st.image(image, caption=f"{source} | {reference}", use_column_width=True)

    with result_column:
        st.markdown(f"## {diagnosis['signal']}")
        st.write(diagnosis["summary"])
        st.success(f"Prioridade recomendada: **{diagnosis['focus']}**")

        metric_density, metric_confidence, metric_priority = st.columns(3)
        metric_density.metric("Padrão", prediction["predicted_class"].title())
        metric_confidence.metric("Confiança", f"{diagnosis['confidence']:.1%}")
        metric_priority.metric("Prioridade", diagnosis["priority"])
        st.caption(diagnosis["contextual_note"])

    st.markdown("### Próximos passos")
    action_column, data_column = st.columns(2, gap="large")
    with action_column:
        with st.container(border=True):
            st.markdown("#### Plano de ação inicial")
            for action in diagnosis["actions"]:
                st.markdown(f"- {action}")
    with data_column:
        with st.container(border=True):
            st.markdown("#### Evidências para confirmar")
            for validation_item in diagnosis["validation_data"]:
                st.markdown(f"- {validation_item}")

    with st.expander("Riscos que justificam a investigação"):
        for risk in diagnosis["risks"]:
            st.markdown(f"- {risk}")

    with st.expander("Detalhes técnicos da classificação"):
        st.caption(f"Modelo: {prediction['model_name']}")
        st.bar_chart(
            pd.DataFrame([prediction["probabilities"]]).T.rename(
                columns={0: "probabilidade"}
            )
        )

    st.caption(
        "Este resultado é uma triagem. Densidade visual não equivale a densidade "
        "populacional e deve ser cruzada com dados locais antes de orientar investimentos."
    )


st.markdown(
    """
    <div class="product-nav">
      <div class="brand-mark">UL</div>
      <div class="brand-copy">
        <strong>UrbanLens Mobility</strong>
        <span>Inteligência territorial para mobilidade</span>
      </div>
      <div class="product-pill">Triagem para decisão pública</div>
    </div>
    <div class="product-hero">
      <div>
        <div class="eyebrow">Planejamento urbano orientado por dados</div>
        <h1>Descubra onde a mobilidade precisa de atenção.</h1>
        <p>
          Analise a ocupação urbana por satélite e receba prioridades e próximos passos
          para planejar a mobilidade.
        </p>
      </div>
      <div class="hero-outcome">
        <span>Da imagem à decisão</span>
        <strong>Território analisado, risco organizado e próximos passos em um único fluxo.</strong>
      </div>
    </div>
    <div class="impact-grid">
      <div class="impact-item">
        <strong>2,5x</strong>
        <span>crescimento da área urbanizada desde 1985</span>
        <small class="impact-source">MapBiomas, 1985–2024</small>
      </div>
      <div class="impact-item">
        <strong>70 mil ha</strong>
        <span>adicionados por ano às cidades brasileiras</span>
        <small class="impact-source">MapBiomas, 1985–2024</small>
      </div>
      <div class="impact-item">
        <strong>1,3 milhão</strong>
        <span>leva mais de 2 horas até o trabalho</span>
        <small class="impact-source">IBGE, Censo 2022</small>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "prediction_cache" not in st.session_state:
    st.session_state.prediction_cache = {}
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

if not MANIFEST_PATH.exists() or not COMPARISON_PATH.exists():
    st.error("Artefatos não encontrados. Rode primeiro `python src/prepare_dataset.py` e `python src/train.py`.")
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

diagnosis_tab, history_tab, methodology_tab = st.tabs(
    ["Diagnóstico", "Relatórios", "Sobre o modelo"]
)

with diagnosis_tab:
    st.markdown(
        """
        <div class="section-kicker">Nova análise</div>
        <h2 class="section-title">Analise um território</h2>
        <p class="section-copy">
          Envie uma imagem ou busque um recorte para gerar prioridade e próximos passos.
        </p>
        """,
        unsafe_allow_html=True,
    )

    input_column, guidance_column = st.columns([1.45, 0.55], gap="large")

    with input_column:
        with st.container(border=True):
            source_mode = st.radio(
                "Como deseja analisar?",
                options=["upload", "territory"],
                format_func=lambda value: {
                    "upload": "Enviar imagem",
                    "territory": "Buscar território",
                }[value],
                horizontal=True,
            )
            transport_supply = st.selectbox(
                "Oferta atual de transporte coletivo",
                options=["Nao informada", "Baixa", "Media", "Alta"],
                format_func=lambda value: {
                    "Nao informada": "Não informada",
                    "Baixa": "Baixa",
                    "Media": "Média",
                    "Alta": "Alta",
                }[value],
                help="Contexto informado pelo usuário para refinar a prioridade.",
            )

            if source_mode == "upload":
                uploaded_file = st.file_uploader(
                    "Imagem aérea ou de satélite",
                    type=["png", "jpg", "jpeg"],
                    help="Use uma imagem com visão superior e boa nitidez.",
                )
                upload_submitted = st.button(
                    "Gerar diagnóstico",
                    type="primary",
                    disabled=uploaded_file is None,
                    use_container_width=True,
                )
                if upload_submitted and uploaded_file is not None:
                    uploaded_bytes = uploaded_file.getvalue()
                    upload_hash = hashlib.sha256(uploaded_bytes).hexdigest()
                    st.session_state.last_analysis = {
                        "image_bytes": uploaded_bytes,
                        "analysis_key": f"upload:{upload_hash}:{transport_supply}",
                        "source": "Upload",
                        "reference": uploaded_file.name,
                        "details": {"oferta_transporte": transport_supply},
                        "transport_supply": transport_supply,
                    }
            else:
                preset_options = {example.name: example for example in DEFAULT_EXAMPLES}
                selected_preset = st.selectbox(
                    "Território de referência",
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
                    location_note = "Informe o centro e a extensão do território."
                else:
                    preset = preset_options[selected_preset]
                    default_latitude = preset.latitude
                    default_longitude = preset.longitude
                    default_bbox_size = preset.bbox_size_meters
                    location_note = preset.note

                st.caption(location_note)
                with st.form("territory_fetch_form"):
                    coordinate_left, coordinate_right = st.columns(2)
                    latitude = coordinate_left.number_input(
                        "Latitude",
                        value=float(default_latitude),
                        format="%.6f",
                    )
                    longitude = coordinate_right.number_input(
                        "Longitude",
                        value=float(default_longitude),
                        format="%.6f",
                    )
                    bbox_size_meters = st.slider(
                        "Extensão analisada (metros)",
                        min_value=2000,
                        max_value=30000,
                        step=1000,
                        value=int(default_bbox_size),
                    )
                    submitted = st.form_submit_button(
                        "Buscar e gerar diagnóstico",
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
                        image_bytes, _, _ = download_arcgis_world_imagery(
                            latitude=latitude,
                            longitude=longitude,
                            bbox_size_meters=bbox_size_meters,
                            output_path=None,
                            image_size=1024,
                        )
                        image_hash = hashlib.sha256(image_bytes).hexdigest()
                        st.session_state.last_analysis = {
                            "image_bytes": image_bytes,
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
                            "transport_supply": transport_supply,
                        }
                    except Exception as error:
                        st.error(f"Não foi possível obter a imagem: {error}")

    with guidance_column:
        with st.container(border=True):
            st.markdown("### Você recebe")
            st.markdown(
                """
                <ul class="mini-list">
                  <li>Padrão visual do território</li>
                  <li>Prioridade de investigação</li>
                  <li>Plano de ação inicial</li>
                  <li>Dados que precisam ser confirmados</li>
                </ul>
                """,
                unsafe_allow_html=True,
            )
        st.info(
            "O sistema faz triagem territorial. A decisão final deve combinar transporte, "
            "população, uso do solo e pesquisa de campo."
        )

    if st.session_state.last_analysis:
        st.divider()
        last_analysis = st.session_state.last_analysis
        run_prediction(
            image=Image.open(io.BytesIO(last_analysis["image_bytes"])),
            analysis_key=last_analysis["analysis_key"],
            source=last_analysis["source"],
            reference=last_analysis["reference"],
            details=last_analysis["details"],
            transport_supply=last_analysis["transport_supply"],
        )
    else:
        st.caption("O último diagnóstico permanecerá aqui durante a sessão.")

with history_tab:
    st.markdown("## Carteira de territórios analisados")
    st.write(
        "Compare os sinais encontrados durante esta sessão e exporte um documento para "
        "discussão técnica ou apresentação."
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
                "referencia": "Território",
                "classe_prevista": "Padrão visual",
                "confianca": "Confiança",
                "sinal_mobilidade": "Sinal territorial",
                "prioridade": "Prioridade",
            }
        )
        st.dataframe(history_preview.iloc[::-1], use_container_width=True, hide_index=True)

        export_column, clear_column = st.columns([3, 1])
        with export_column:
            st.download_button(
                "Exportar relatório executivo",
                data=build_analysis_report(st.session_state.analysis_history),
                file_name=f"urbanlens_mobility_{date.today().isoformat()}.html",
                mime="text/html",
                type="primary",
                use_container_width=True,
            )
        with clear_column:
            if st.button("Limpar histórico", use_container_width=True):
                st.session_state.analysis_history = []
                st.session_state.prediction_cache = {}
                st.session_state.last_analysis = None
                st.rerun()
    else:
        st.info("As análises feitas nesta sessão aparecerão aqui.")
        st.download_button(
            "Exportar relatório executivo",
            data=b"",
            file_name=f"urbanlens_mobility_{date.today().isoformat()}.html",
            mime="text/html",
            disabled=True,
            use_container_width=True,
        )
        st.caption("Analise ao menos um território para liberar o relatório.")

with methodology_tab:
    st.markdown("## Metodologia e limites")
    st.warning(
        "O modelo classifica densidade urbana visual. Ele não estima diretamente população, "
        "renda, demanda de viagens, congestionamento ou qualidade da infraestrutura."
    )

    model_column, validation_column, use_column = st.columns(3)
    model_column.metric("Modelo em produção", best_model_name)
    validation_column.metric(
        "Melhor acurácia de validação",
        f"{metadata['best_validation_accuracy'] * 100:.2f}%",
    )
    use_column.metric("Acurácia em teste", f"{metadata['test_accuracy'] * 100:.2f}%")

    with st.expander("Desempenho e comparação dos modelos"):
        st.dataframe(comparison, use_container_width=True)
        visual_left, visual_right = st.columns(2)
        with visual_left:
            st.markdown("#### Matriz de confusão")
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
                    column.warning(f"Imagem indisponível: {caption}")

    st.markdown("### Referências de planejamento")
    st.markdown(
        """
        - [WRI Brasil - Desenvolvimento Orientado ao Transporte Sustentável](https://www.wribrasil.org.br/projetos/desenvolvimento-orientado-ao-transporte-sustentavel-dots)
        - [ITDP - Princípios de Desenvolvimento Orientado ao Transporte](https://itdp.org/publication/principios-desenvolvimento-orientado-ao-transporte/)
        - [UN-Habitat - Planejamento e desenho urbano](https://unhabitat.org/planning-and-design/)
        """
    )
    st.caption(f"Artefato do modelo em uso: {best_artifact_path}")
