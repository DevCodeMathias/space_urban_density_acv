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
            "A ocupação visual dispersa pode indicar distâncias maiores entre moradia, "
            "trabalho e serviços, elevando o custo do transporte e a dependência do automóvel."
        ),
        "risks": [
            "Baixa viabilidade do transporte coletivo convencional.",
            "Longos deslocamentos para acessar empregos e serviços.",
            "Maior custo de infraestrutura por habitante.",
        ],
        "actions": [
            "Mapear centralidades e destinos essenciais pouco conectados.",
            "Avaliar linhas alimentadoras, transporte sob demanda e integração tarifária.",
            "Priorizar rotas seguras de caminhada e bicicleta.",
            "Coordenar novas ocupações com corredores de transporte existentes.",
        ],
        "validation_data": [
            "Tempo médio de viagem e matriz origem-destino.",
            "Cobertura e frequência do transporte coletivo.",
            "Distância a empregos, escolas, saúde e comércio.",
            "Evolução da mancha urbana e do uso do solo.",
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
            "Expansão futura sem conexão com transporte.",
            "Dependência do automóvel em bairros monofuncionais.",
            "Desigualdade de acesso entre setores do território.",
        ],
        "actions": [
            "Preservar a conexão entre bairros e corredores de transporte.",
            "Fortalecer linhas estruturais e integração com caminhada e bicicleta.",
            "Estimular serviços de proximidade nos eixos consolidados.",
            "Monitorar o crescimento para evitar espraiamento ou saturação.",
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
            "A ocupação visual concentrada pode sustentar transporte de alta capacidade, "
            "mas exige infraestrutura compatível para evitar saturação e exclusão."
        ),
        "risks": [
            "Lotação e baixa confiabilidade do transporte coletivo.",
            "Conflitos entre pedestres, bicicletas, ônibus e automóveis.",
            "Barreiras de acesso em áreas muito concentradas.",
        ],
        "actions": [
            "Medir lotação e ampliar capacidade nos corredores de maior demanda.",
            "Qualificar calçadas, travessias e acesso aos pontos de transporte.",
            "Integrar bicicleta, linhas alimentadoras e transporte de alta capacidade.",
            "Reorganizar estacionamento, carga e descarga e circulação viária.",
        ],
        "validation_data": [
            "Lotação por faixa horária e velocidade operacional.",
            "Fluxo de pedestres e segurança viária.",
            "Cobertura de calçadas, ciclovias e áreas de integração.",
            "Capacidade dos serviços públicos locais.",
        ],
    },
}

st.set_page_config(page_title=PRODUCT_NAME, layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid #d8e2dc;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        background: #ffffff;
    }
    div[data-testid="stFileUploaderDropzone"] {
        border-radius: 10px;
    }
    div[data-testid="stImage"] img {
        border-radius: 10px;
    }
    @media (max-width: 720px) {
        .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_manifest() -> pd.DataFrame:
    return pd.read_csv(MANIFEST_PATH)


@st.cache_data
def load_comparison() -> pd.DataFrame:
    return pd.read_csv(COMPARISON_PATH)


def resolve_project_path(relative_path: str) -> Path:
    normalized_path = str(relative_path).replace("\\", "/")
    return PROJECT_ROOT.joinpath(*normalized_path.split("/"))


def build_mobility_diagnosis(prediction: dict, transport_supply: str) -> dict:
    predicted_class = prediction["predicted_class"]
    confidence = max(prediction["probabilities"].values())
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
        "Baixa": "A baixa oferta informada reforça a necessidade de investigar lacunas de acesso.",
        "Media": "Compare a oferta informada com demanda, frequência e cobertura.",
        "Alta": "A oferta é positiva, mas capacidade e acesso de última milha ainda precisam ser medidos.",
        "Nao informada": "Cruze o resultado com a oferta de transporte para refinar a prioridade.",
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
        "oferta_transporte": "Não informada",
    }
    record.update(details or {})
    st.session_state.analysis_history.append(record)


def build_analysis_report(history: list[dict]) -> bytes:
    report_frame = pd.DataFrame(history).copy()

    def percentage(value) -> str:
        try:
            return f"{float(value):.2%}"
        except (TypeError, ValueError):
            return ""

    expected_columns = [
        "data_hora_utc",
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
        "area_metros",
    ]
    for column in expected_columns:
        if column not in report_frame:
            report_frame[column] = ""
    report_frame = report_frame[expected_columns]
    report_frame["confianca"] = report_frame["confianca"].map(percentage)
    for column in ("prob_baixa", "prob_media", "prob_alta"):
        report_frame[column] = report_frame[column].map(percentage)

    report_frame = report_frame.rename(
        columns={
            "data_hora_utc": "Data/hora UTC",
            "referencia": "Território",
            "classe_prevista": "Padrão visual",
            "confianca": "Confiança",
            "sinal_mobilidade": "Sinal territorial",
            "prioridade": "Prioridade",
            "foco": "Foco recomendado",
            "oferta_transporte": "Oferta de transporte",
            "acoes_recomendadas": "Ações recomendadas",
            "dados_para_validar": "Dados para validar",
            "prob_baixa": "Prob. baixa",
            "prob_media": "Prob. média",
            "prob_alta": "Prob. alta",
            "modelo": "Modelo",
            "latitude": "Latitude",
            "longitude": "Longitude",
            "area_metros": "Área (m)",
        }
    )
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <title>Relatório UrbanLens Mobility</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #172033; }}
    h1, h2 {{ color: #075e54; }}
    .summary {{ background: #eef6f2; padding: 16px; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; font-size: 12px; }}
    th, td {{ border: 1px solid #ccd8d2; padding: 8px; text-align: left; }}
    th {{ background: #e4efe9; }}
    tr:nth-child(even) {{ background: #f8faf9; }}
    .footer {{ color: #627d98; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>UrbanLens Mobility</h1>
  <div class="summary">
    <strong>{len(history)} território(s) analisado(s)</strong><br>
    Relatório gerado em {generated_at}.
  </div>
  <h2>Diagnósticos e recomendações</h2>
  {report_frame.to_html(index=False, border=0, escape=True)}
  <h2>Limite de uso</h2>
  <p>
    O resultado representa densidade urbana visual. Ele deve ser validado com dados
    populacionais, transporte, uso do solo e pesquisa de campo.
  </p>
  <p class="footer">Relatório gerado pelo {PRODUCT_NAME}.</p>
</body>
</html>
"""
    return html.encode("utf-8")


def render_prediction(
    image: Image.Image,
    analysis_key: str,
    source: str,
    reference: str,
    transport_supply: str,
    details: dict | None = None,
) -> None:
    if analysis_key in st.session_state.prediction_cache:
        prediction = st.session_state.prediction_cache[analysis_key]
    else:
        prediction = predict_image(image)
        st.session_state.prediction_cache[analysis_key] = prediction
        diagnosis = build_mobility_diagnosis(prediction, transport_supply)
        record_analysis(prediction, diagnosis, source, reference, details)

    diagnosis = build_mobility_diagnosis(prediction, transport_supply)
    image_column, result_column = st.columns([0.85, 1.15])

    with image_column:
        st.image(image, caption=f"{source} | {reference}", use_column_width=True)

    with result_column:
        st.subheader(diagnosis["signal"])
        st.write(diagnosis["summary"])
        st.success(f"Foco recomendado: **{diagnosis['focus']}**")

        density_metric, confidence_metric, priority_metric = st.columns(3)
        density_metric.metric("Padrão", prediction["predicted_class"].title())
        confidence_metric.metric("Confiança", f"{diagnosis['confidence']:.1%}")
        priority_metric.metric("Prioridade", diagnosis["priority"])
        st.caption(diagnosis["contextual_note"])

    action_column, evidence_column = st.columns(2)
    with action_column:
        st.markdown("#### Plano de ação inicial")
        for action in diagnosis["actions"]:
            st.markdown(f"- {action}")
    with evidence_column:
        st.markdown("#### Dados para validar")
        for validation_item in diagnosis["validation_data"]:
            st.markdown(f"- {validation_item}")

    with st.expander("Ver riscos e detalhes do modelo"):
        st.markdown("**Riscos que justificam a investigação**")
        for risk in diagnosis["risks"]:
            st.markdown(f"- {risk}")
        st.caption(f"Modelo: {prediction['model_name']}")
        st.bar_chart(
            pd.DataFrame([prediction["probabilities"]]).T.rename(
                columns={0: "probabilidade"}
            )
        )

    st.caption(
        "Triagem de apoio à decisão: densidade visual não equivale a densidade "
        "populacional e não substitui dados locais."
    )


if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "prediction_cache" not in st.session_state:
    st.session_state.prediction_cache = {}
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

if not MANIFEST_PATH.exists() or not COMPARISON_PATH.exists():
    st.error(
        "Artefatos não encontrados. Rode `python src/prepare_dataset.py` "
        "e `python src/train.py`."
    )
    st.stop()

metadata = load_metadata()
comparison = load_comparison()
manifest = load_manifest()
best_model_name = metadata["best_model"]
confusion_matrix_path = (
    PROJECT_ROOT / "reports" / f"confusion_matrix_{best_model_name}.png"
)
error_examples_path = (
    PROJECT_ROOT / "reports" / f"error_examples_{best_model_name}.png"
)

st.title("UrbanLens Mobility")
st.caption(
    "Diagnóstico visual de densidade urbana para apoiar o planejamento de mobilidade."
)
st.info(
    "A área urbanizada brasileira cresceu 2,5 vezes desde 1985. "
    "Use a análise para identificar onde aprofundar estudos de transporte e acesso."
)

input_column, context_column = st.columns([1.15, 0.85])

with input_column:
    st.subheader("Analisar território")
    source_mode = st.radio(
        "Fonte da imagem",
        options=["upload", "territory"],
        format_func=lambda value: {
            "upload": "Enviar imagem",
            "territory": "Buscar por coordenadas",
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
        help="Contexto opcional usado para refinar a prioridade.",
    )

    if source_mode == "upload":
        uploaded_file = st.file_uploader(
            "Envie uma imagem aérea ou de satélite",
            type=["png", "jpg", "jpeg"],
        )
        upload_submitted = st.button(
            "Gerar diagnóstico",
            type="primary",
            disabled=uploaded_file is None,
            use_container_width=True,
        )
        if upload_submitted and uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            image_hash = hashlib.sha256(image_bytes).hexdigest()
            st.session_state.last_analysis = {
                "image_bytes": image_bytes,
                "analysis_key": f"upload:{image_hash}:{transport_supply}",
                "source": "Upload",
                "reference": uploaded_file.name,
                "transport_supply": transport_supply,
                "details": {"oferta_transporte": transport_supply},
            }
    else:
        preset_options = {example.name: example for example in DEFAULT_EXAMPLES}
        selected_preset = st.selectbox(
            "Local de referência",
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
            latitude = st.number_input(
                "Latitude", value=float(default_latitude), format="%.6f"
            )
            longitude = st.number_input(
                "Longitude", value=float(default_longitude), format="%.6f"
            )
            bbox_size_meters = st.slider(
                "Área capturada ao redor do ponto (metros)",
                min_value=2000,
                max_value=30000,
                step=1000,
                value=int(default_bbox_size),
            )
            submitted = st.form_submit_button(
                "Baixar imagem e analisar",
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
                st.caption(f"Fonte: [ArcGIS World Imagery]({source_url})")
                st.session_state.last_analysis = {
                    "image_bytes": image_bytes,
                    "analysis_key": f"satellite:{image_hash}:{transport_supply}",
                    "source": "ArcGIS World Imagery",
                    "reference": request_name,
                    "transport_supply": transport_supply,
                    "details": {
                        "latitude": latitude,
                        "longitude": longitude,
                        "area_metros": bbox_size_meters,
                        "oferta_transporte": transport_supply,
                    },
                }
            except Exception as error:
                st.error(f"Não foi possível obter a imagem: {error}")

with context_column:
    st.subheader("Por que isso importa")
    impact_one, impact_two = st.columns(2)
    impact_one.metric("Expansão urbana desde 1985", "2,5x")
    impact_two.metric("Viagens acima de 2 horas", "1,3 mi")
    st.caption("Fontes: MapBiomas 1985–2024 e IBGE, Censo 2022.")

    st.markdown("#### O resultado entrega")
    st.markdown(
        """
        - padrão visual de ocupação;
        - prioridade de investigação;
        - plano de ação inicial;
        - dados necessários para validar a hipótese.
        """
    )
    st.warning(
        "A ferramenta faz triagem territorial. A decisão final deve combinar "
        "população, transporte, uso do solo e pesquisa de campo."
    )
    st.metric("Acurácia atual em teste", f"{metadata['test_accuracy'] * 100:.2f}%")

if st.session_state.last_analysis:
    st.divider()
    st.subheader("Resultado da análise")
    last_analysis = st.session_state.last_analysis
    render_prediction(
        image=Image.open(io.BytesIO(last_analysis["image_bytes"])),
        analysis_key=last_analysis["analysis_key"],
        source=last_analysis["source"],
        reference=last_analysis["reference"],
        transport_supply=last_analysis["transport_supply"],
        details=last_analysis["details"],
    )

st.divider()
st.subheader("Últimas análises")
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
            "classe_prevista": "Padrão",
            "confianca": "Confiança",
            "sinal_mobilidade": "Sinal territorial",
            "prioridade": "Prioridade",
        }
    )
    st.dataframe(history_preview.iloc[::-1], use_container_width=True, hide_index=True)

    export_column, clear_column = st.columns([3, 1])
    with export_column:
        st.download_button(
            "Exportar relatório das análises",
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
    st.info("As análises realizadas nesta sessão aparecerão aqui.")
    st.download_button(
        "Exportar relatório das análises",
        data=b"",
        file_name=f"urbanlens_mobility_{date.today().isoformat()}.html",
        mime="text/html",
        disabled=True,
        use_container_width=True,
    )

with st.expander("Metodologia, desempenho e amostras"):
    st.warning(
        "O modelo classifica densidade urbana visual. Ele não estima diretamente "
        "população, renda, demanda de viagens ou qualidade da infraestrutura."
    )
    model_metric, validation_metric, test_metric = st.columns(3)
    model_metric.metric("Modelo", best_model_name)
    validation_metric.metric(
        "Acurácia de validação",
        f"{metadata['best_validation_accuracy'] * 100:.2f}%",
    )
    test_metric.metric("Acurácia em teste", f"{metadata['test_accuracy'] * 100:.2f}%")
    st.dataframe(comparison, use_container_width=True)

    visual_left, visual_right = st.columns(2)
    with visual_left:
        if confusion_matrix_path.exists():
            st.image(
                str(confusion_matrix_path),
                caption="Matriz de confusão",
                use_column_width=True,
            )
    with visual_right:
        if error_examples_path.exists():
            st.image(
                str(error_examples_path),
                caption="Exemplos de erros",
                use_column_width=True,
            )

    st.markdown("#### Amostras do dataset")
    sample_rows = manifest.groupby("visual_density_class").head(2).reset_index(drop=True)
    for start_index in range(0, len(sample_rows), 3):
        sample_columns = st.columns(3)
        for column, (_, row) in zip(
            sample_columns,
            sample_rows.iloc[start_index : start_index + 3].iterrows(),
        ):
            image_path = resolve_project_path(row["local_image_path"])
            caption = f"{row['visual_density_class']} | {row['zone_type']}"
            if image_path.is_file():
                column.image(str(image_path), caption=caption, use_column_width=True)

with st.expander("Referências utilizadas"):
    st.markdown(
        """
        - [MapBiomas — expansão urbana brasileira](https://brasil.mapbiomas.org/)
        - [IBGE — deslocamentos para o trabalho no Censo 2022](https://www.ibge.gov.br/)
        - [WRI Brasil — Desenvolvimento Orientado ao Transporte](https://www.wribrasil.org.br/projetos/desenvolvimento-orientado-ao-transporte-sustentavel-dots)
        - [ITDP — Princípios de Desenvolvimento Orientado ao Transporte](https://itdp.org/publication/principios-desenvolvimento-orientado-ao-transporte/)
        """
    )
