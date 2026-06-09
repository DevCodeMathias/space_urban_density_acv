# Arquitetura dos modelos

Este documento descreve as arquiteturas treinadas para classificar imagens de
satelite em tres classes de densidade urbana visual: `baixa`, `media` e `alta`.

## Fluxo da solucao

1. A aplicacao recebe uma imagem enviada pelo usuario ou obtida por coordenadas.
2. A imagem RGB e redimensionada para `64 x 64` pixels.
3. Os canais sao normalizados com estatisticas calculadas apenas no treino.
4. A CNN produz logits para as tres classes.
5. `softmax` converte os logits em probabilidades.
6. A classe com maior probabilidade alimenta o diagnostico territorial.

## UrbanDensityCNNV1

Modelo baseline e melhor modelo atual.

```text
Entrada RGB (3 x 64 x 64)
-> Conv2d 3->32 + ReLU + MaxPool
-> Conv2d 32->64 + ReLU + MaxPool
-> Conv2d 64->128 + ReLU + MaxPool
-> Flatten
-> Linear 8192->256 + ReLU + Dropout(0,35)
-> Linear 256->3
```

- parametros treinaveis: `2.191.427`
- acuracia de teste: `88,89%`
- pesos: `models/best_model.pt` e `models/urban_cnn_v1.pt`

## UrbanDensityCNNV2

CNN compacta com blocos convolucionais duplos.

```text
Entrada RGB
-> ConvBlock 3->32
-> ConvBlock 32->64
-> ConvBlock 64->128
-> AdaptiveAvgPool 1 x 1
-> Linear 128->96 + ReLU + Dropout(0,35)
-> Linear 96->3
```

Cada `ConvBlock` possui duas convolucoes, `BatchNorm`, `ReLU` e `MaxPool`.

- parametros treinaveis: `300.579`
- pesos: `models/urban_cnn_v2.pt`

## UrbanDensityCNNV3

CNN residual para extracao de representacoes mais profundas.

```text
Entrada RGB
-> Stem Conv2d 3->32 + BatchNorm + ReLU
-> 7 blocos residuais com canais 32, 64, 128 e 192
-> AdaptiveAvgPool 1 x 1
-> Linear 192->128 + ReLU + Dropout(0,40)
-> Linear 128->3
```

- parametros treinaveis: `1.944.867`
- pesos: `models/urban_cnn_v3.pt`

## UrbanDensityCNNV4

CNN residual multitarefa. Alem da classificacao, possui uma cabeca auxiliar que
estima a proporcao visual de area construida.

```text
Entrada RGB
-> Stem e 7 blocos residuais
-> AdaptiveAvgPool 1 x 1
-> Embedding 192->160
-> Cabeca auxiliar 160->64->1
-> Concatenacao embedding + proporcao construida
-> Classificador 161->128->3
```

- parametros treinaveis: `1.982.148`
- pesos: `models/urban_cnn_v4.pt`

## Modelo de stacking

O `UrbanMetaStackV1` combina probabilidades produzidas pelas CNNs e atributos
visuais calculados sobre a imagem. O artefato esta em
`models/urban_meta_stack_v1.joblib`.

## Treinamento

- funcao de perda principal: `CrossEntropyLoss`
- otimizador: `AdamW`
- scheduler: `ReduceLROnPlateau`
- regularizacao: `Dropout`, augmentations e early stopping
- epocas maximas: `45`
- paciencia do early stopping: `12`
- seed: `42`

## Arquivos de referencia

- implementacao executavel: `src/models.py`
- treinamento e avaliacao: `src/train.py`
- arquitetura serializada: `reports/model_architectures.json`
- resumo textual gerado: `reports/model_architectures.txt`
- comparacao de desempenho: `reports/model_comparison.csv`
