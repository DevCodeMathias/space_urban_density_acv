# Applied Computer Vision - Urban Density from Satellite Imagery

Projeto de Visao Computacional aplicado a Industria Espacial e conectado ao tema da Global Solution da equipe: monitoramento de densidade urbana a partir de imagens de satelite.

## Integrantes

- Integrante 1 - preencher nome
- Integrante 2 - preencher nome
- Integrante 3 - preencher nome
- Integrante 4 - preencher nome

## 1. Definicao do problema de visao computacional

O problema resolvido neste projeto e a **classificacao de densidade urbana visual** em imagens reais de satelite.

Cada imagem e classificada em uma de tres categorias:

- `baixa`
- `media`
- `alta`

Essa etapa complementa a Global Solution anterior da equipe. No projeto integrado, a densidade urbana foi tratada por Machine Learning tabular; aqui, o objetivo foi adicionar uma camada de **Visao Computacional**, permitindo que a ocupacao urbana seja estimada a partir do padrao visual observado em imagens orbitais.

No contexto da Industria Espacial, a proposta se conecta ao uso de observacao da Terra e sensoriamento remoto para:

- monitoramento de expansao urbana
- apoio ao planejamento de infraestrutura
- identificacao de regioes mais adensadas
- suporte a analises espaciais automatizadas

## 2. Dataset utilizado

O dataset foi **criado e adaptado pela equipe** a partir de recortes reais baixados pelo ArcGIS World Imagery.

### Origem das imagens

- imagens reais de satelite baixadas pelo servico ArcGIS World Imagery
- camada utilizada: `World_Imagery`
- recortes RGB salvos em `data/raw_images/`
- cenas reais com diferentes niveis de area construida, vegetacao, agua e vias

### Classes utilizadas

As classes finais foram definidas como uma medida de **densidade urbana visual**, derivada do nivel de ocupacao construido presente na cena:

- `baixa`
- `media`
- `alta`

Essa formulacao foi escolhida porque e mais aderente ao conteudo efetivamente visivel da imagem, o que torna o problema mais coerente com a proposta de Visao Computacional.

### Quantidade de imagens

- total: `45`
- `baixa`: `15`
- `media`: `15`
- `alta`: `15`

### Divisao entre treino, validacao e teste

A divisao foi feita de forma estratificada:

- treino: `31` imagens
- validacao: `7` imagens
- teste: `7` imagens

Distribuicao final por split:

- treino: `10 baixa`, `10 media`, `11 alta`
- validacao: `2 baixa`, `3 media`, `2 alta`
- teste: `3 baixa`, `2 media`, `2 alta`

Arquivo de distribuicao:

- `reports/class_distribution.csv`

### Pre-processamento

As principais etapas foram:

- download dos recortes reais pelo ArcGIS World Imagery
- organizacao do dataset em projeto independente
- split estratificado treino, validacao e teste
- redimensionamento para `64 x 64`
- normalizacao com media e desvio padrao calculados no conjunto de treino
- data augmentation no treino com:
- `RandomHorizontalFlip`
- `RandomRotation`
- `ColorJitter`

Estatisticas salvas em:

- `reports/normalization_stats.json`

## 3. Treinamento de CNNs do zero

Foram implementadas duas arquiteturas proprias, **sem uso de modelos pre-treinados**.

### Modelo 1 - `UrbanDensityCNNV1`

Arquitetura baseline composta por:

- 3 camadas convolucionais
- `ReLU`
- `MaxPooling`
- camada totalmente conectada com `Dropout`

Parametros treinaveis:

- `2.191.427`

### Modelo 2 - `UrbanDensityCNNV2`

Arquitetura mais eficiente e regularizada, composta por:

- blocos convolucionais duplos
- `BatchNorm`
- `ReLU`
- `MaxPooling`
- `AdaptiveAvgPool`
- camada densa final com `Dropout`

Parametros treinaveis:

- `300.579`

### Estrategia de treinamento

- `CrossEntropyLoss`
- pesos por classe
- `AdamW`
- `ReduceLROnPlateau`
- `early stopping`

Arquivos de arquitetura:

- `reports/model_architectures.json`
- `reports/model_architectures.txt`
- `src/models.py`

## 4. Avaliacao dos modelos

As metricas usadas para avaliacao foram:

- acuracia
- loss
- matriz de confusao
- classification report
- analise de erros
- curvas de treino e validacao

### Resultados quantitativos

| Modelo | Parametros | Melhor acuracia de validacao | Acuracia em teste | Loss em teste |
|---|---:|---:|---:|---:|
| UrbanDensityCNNV1 | 2.191.427 | 85,71% | 71,43% | 0,4787 |
| UrbanDensityCNNV2 | 300.579 | 42,86% | 28,57% | 1,0961 |
| UrbanDensityCNNV3 | 1.944.867 | 71,43% | 42,86% | 1,0582 |
| UrbanDensityCNNV4 | 1.982.148 | 71,43% | 42,86% | 1,1622 |
| UrbanMetaStackV1 | 60 | 100,00% | 42,86% | n/a |

### Melhor modelo

- melhor arquitetura atual: `UrbanDensityCNNV1`
- acuracia final em teste: `71,43%`

Arquivos gerados:

- `reports/model_comparison.csv`
- `reports/summary.json`
- `models/urban_cnn_v1.pt`
- `models/urban_cnn_v2.pt`
- `models/best_model.pt`
- `models/best_model_metadata.json`

### Avaliacao qualitativa

Tambem foram gerados:

- matrizes de confusao para os dois modelos
- exemplos de imagens classificadas incorretamente
- classification report por classe

Arquivos:

- `reports/confusion_matrix_urban_cnn_v1.png`
- `reports/confusion_matrix_urban_cnn_v2.png`
- `reports/error_examples_urban_cnn_v1.png`
- `reports/error_examples_urban_cnn_v2.png`
- `reports/misclassifications_urban_cnn_v1.csv`
- `reports/misclassifications_urban_cnn_v2.csv`
- `reports/classification_report_urban_cnn_v1.json`
- `reports/classification_report_urban_cnn_v2.json`

## 5. Comparacao entre arquiteturas

A comparacao mostra uma diferenca tecnica clara entre os modelos.

### `UrbanDensityCNNV1`

- tem muito mais parametros
- usa um bloco mais simples, sem BatchNorm
- apresenta maior propensao a oscilacao entre treino e validacao

### `UrbanDensityCNNV2`

- usa menos parametros
- possui blocos convolucionais mais estruturados
- usa `BatchNorm`, `AdaptiveAvgPool` e regularizacao mais consistente
- generalizou melhor no conjunto de teste

### Conclusao tecnica

Mesmo com menos parametros, `UrbanDensityCNNV2` foi superior porque sua arquitetura extraiu padroes espaciais de forma mais estavel e com menor risco de overfitting do que a baseline.

## 6. Meta de acuracia e justificativa tecnica

O enunciado define uma referencia minima de **88% de acuracia no conjunto de teste**.

Resultado obtido pelo melhor modelo no dataset real atual:

- `71,43%`

### Justificativa tecnica

A meta de 88% nao foi atingida. A principal justificativa tecnica e:

- o dataset real atual e pequeno (`45` imagens)
- as imagens sao pequenas (`64 x 64`) e bastante ruidosas
- existe sobreposicao visual entre classes medias e altas
- diferentes zonas urbanas reais compartilham texturas parecidas
- parte da separacao entre classes ainda depende de sinais sutis, como mistura entre vegetacao, vias e area construida

### Melhorias futuras

- aumentar a resolucao das imagens
- gerar mais imagens por classe
- refinar ainda mais os limiares das classes visuais
- testar schedules de learning rate mais agressivos
- aplicar augmentations adicionais
- incluir interpretabilidade visual com Grad-CAM

## 7. Demonstracao funcional

A demonstracao funcional foi implementada em **Streamlit**.

A interface permite:

- enviar uma nova imagem
- obter a classe prevista
- visualizar probabilidades por classe
- consultar a comparacao dos modelos
- ver matriz de confusao e exemplos de erros

Arquivo principal:

- `app/app.py`

Aplicacao online:

- https://spaceurbandensityacv-4sf4bcwc4gpxrf7mv9zdor.streamlit.app/

Execucao:

```bash
streamlit run app/app.py
```

## 8. Notebook utilizado

O projeto inclui um notebook em formato `.ipynb` com o fluxo de apoio ao treinamento:

- `notebooks/acv_training.ipynb`

## 9. Estrutura do projeto

```text
space_urban_density_acv/
|-- app/
|   |-- app.py
|-- data/
|   |-- manifest.csv
|   |-- raw_images/
|   |-- splits/
|-- models/
|   |-- urban_cnn_v1.pt
|   |-- urban_cnn_v2.pt
|   |-- best_model.pt
|   |-- best_model_metadata.json
|-- notebooks/
|   |-- acv_training.ipynb
|-- reports/
|   |-- class_distribution.csv
|   |-- normalization_stats.json
|   |-- model_architectures.json
|   |-- model_architectures.txt
|   |-- model_comparison.csv
|   |-- confusion_matrix_urban_cnn_v1.png
|   |-- confusion_matrix_urban_cnn_v2.png
|   |-- training_curves_urban_cnn_v1.png
|   |-- training_curves_urban_cnn_v2.png
|   |-- error_examples_urban_cnn_v1.png
|   |-- error_examples_urban_cnn_v2.png
|-- src/
|   |-- config.py
|   |-- data_utils.py
|   |-- inference.py
|   |-- models.py
|   |-- prepare_dataset.py
|   |-- train.py
|-- requirements.txt
|-- README.md
```

## 10. Como executar

### 10.1 Criar ambiente virtual

```bash
python -m venv .venv
```

### 10.2 Ativar ambiente

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

### 10.3 Instalar dependencias

```bash
pip install -r requirements.txt
```

### 10.4 Preparar o dataset

```bash
python src/prepare_dataset.py
```

### 10.5 Treinar as CNNs

```bash
python src/train.py
```

### 10.6 Rodar a demonstracao

```bash
streamlit run app/app.py
```

## 11. Itens exigidos pelo enunciado

Este repositorio contem:

- notebook `.ipynb`
- scripts Python
- arquivo com a arquitetura dos modelos
- pesos do melhor modelo
- imagens do dataset
- `requirements.txt`
- aplicacao funcional
- README com instrucoes
