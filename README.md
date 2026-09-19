# Thetis — Few-Shot Action Recognition para Golpes de Tênis

Pesquisa e implementação de métodos de **Few-Shot Learning (FSL)** aplicados ao reconhecimento de ações específicas em vídeos de tênis. Os experimentos são conduzidos sobre o dataset [THETIS](https://github.com/THETIS-dataset/dataset), adaptado ao protocolo **N-way K-shot** padrão da literatura de Few-Shot Action Recognition (FSAR).

---

## Contexto e motivação

O reconhecimento de ações em vídeo evoluiu fortemente com arquiteturas profundas treinadas em datasets massivos (Kinetics-400, ~400 classes, ~600k vídeos). Em domínios esportivos específicos, no entanto, a anotação é cara e exige especialistas, e classes vizinhas (por exemplo, *forehand topspin* vs *forehand flat*, ou *saque plano* vs *saque slice*) têm baixa variância visual. Datasets de tênis como o Tennis7 possuem poucas dezenas de exemplos por classe, o que torna inviável o paradigma supervisionado tradicional.

Este projeto investiga como métodos de **Few-Shot Action Recognition** se comportam nesse cenário, com atenção a duas dificuldades específicas do tênis:

- variação de velocidade do mesmo golpe entre jogadores de níveis distintos (iniciante, amador, profissional);
- granularidade fina entre classes vizinhas, que compartilham boa parte da cinemática.

A avaliação segue o protocolo **N-way K-shot** consolidado por Snell et al. (Prototypical Networks): o modelo recebe N classes novas com K exemplos rotulados (suporte) e classifica consultas dessas classes.

---

## Perguntas de pesquisa

1. Entre os métodos de FSAR recentes, quais entregam melhor desempenho no reconhecimento de golpes específicos de tênis?
2. Como esses métodos lidam com a variação de velocidade do mesmo golpe entre jogadores de níveis diferentes?
3. A combinação de **pose 2D**, **fluxo óptico** e **descrições textuais** melhora a discriminação de classes de granularidade fina, em comparação com métodos puramente RGB?
4. É possível obter desempenho competitivo com apenas 5 exemplos por classe (5-way 5-shot)?

---

## Métodos avaliados

A comparação cobre famílias representativas de FSAR:

| Família | Método | Referência |
| --- | --- | --- |
| Metric learning (baseline) | Prototypical Networks | Snell et al. |
| Cross-attention temporal | TRX | Perrett et al. |
| Alinhamento multi-velocidade | MVP-Shot | — |
| Multimodal (vídeo + texto) | SAFSAR | Tang et al. |
| Baseado em pose | VPD | Hong et al. |
| Movimento denso | SOAP | — |
| Atenção bidirecional fina | BAM + CML | — |

A intenção não é propor uma arquitetura nova, mas comparar sistematicamente as abordagens no domínio do tênis.

---

## Dataset experimental: THETIS

Os experimentos usam o **THETIS (Three dimEnsional TennIs Shots)**, capturado com Kinect:

- **8.374 sequências de vídeo**
- **55 sujeitos**: p1–p31 iniciantes, p32–p55 especialistas
- **12 classes**: Backhand, Backhand 2 mãos, Backhand slice, Backhand volley, Forehand flat, Forehand open stance, Forehand slice, Forehand volley, Serviço flat, Serviço kick, Serviço slice, Smash
- **5 modalidades**: RGB, Depth, Mask (silhueta), Skeleton 2D, Skeleton 3D

> **Observação metodológica.** O THETIS, em sua forma original, possui centenas de exemplos por classe e portanto não é, por si só, um benchmark few-shot. Neste projeto ele é usado como base experimental: subamostragens controladas das 12 classes geram episódios **N-way K-shot** (tipicamente 5-way 1-shot e 5-way 5-shot) que simulam o cenário de escassez de dados. A divisão de classes entre meta-train, meta-val e meta-test é detalhada em `experiments/configs/`. A inclusão do Tennis7 como dataset complementar para validação cruzada está prevista.

---

## Estrutura do projeto

```text
Thetis/
├── dataset/                    # Clone do repositório THETIS (não versionado)
│   ├── VIDEO_RGB/
│   ├── VIDEO_Depth/
│   ├── VIDEO_Mask/
│   ├── VIDEO_Skelet2D/
│   └── VIDEO_Skelet3D/
│
├── data/                       # Dados processados (não versionados)
│   ├── processed/              # Features extraídas: esqueletos normalizados,
│   │                           # fluxo óptico, embeddings de texto, etc.
│   └── episodes/               # Episódios N-way K-shot pré-amostrados
│       ├── meta_train/
│       ├── meta_val/
│       └── meta_test/
│
├── src/
│   ├── data/
│   │   ├── loader.py           # Parsing das sequências e modalidades
│   │   ├── episode_sampler.py  # Amostragem N-way K-shot (suporte + consulta)
│   │   └── augment.py          # Aumentação espaço-temporal
│   ├── features/
│   │   ├── pose.py             # Extração e normalização de pose 2D
│   │   ├── optical_flow.py     # Cálculo de fluxo óptico
│   │   └── text.py             # Embeddings de descrições textuais dos golpes
│   ├── models/
│   │   ├── protonet.py         # Baseline: Prototypical Networks
│   │   ├── trx.py              # TRX (cross-attention temporal)
│   │   ├── mvp_shot.py         # MVP-Shot (alinhamento multi-velocidade)
│   │   ├── safsar.py           # SAFSAR (vídeo + texto)
│   │   └── vpd.py              # Video Pose Distillation
│   ├── training/
│   │   ├── meta_trainer.py     # Loop de meta-treino episódico
│   │   └── eval_episodic.py    # Avaliação N-way K-shot
│   └── utils/
│       ├── metrics.py          # Acurácia média por episódio, IC 95%, F1
│       └── viz.py              # Visualização de episódios e confusões
│
├── notebooks/
│   ├── 01_eda.ipynb            # Análise exploratória do THETIS
│   ├── 02_episode_design.ipynb # Construção dos splits episódicos
│   ├── 03_results.ipynb        # Análise comparativa entre métodos
│   └── kaggle_train_protonet.ipynb  # Meta-treino no Kaggle (5 modalidades × 1/5-shot, com --resume)
│
├── experiments/
│   ├── configs/                # Um .yaml por experimento (método × modalidade × N × K)
│   │   └── README.md           # Referência das chaves de config (defaults, VRAM, limitações)
│   └── logs/                   # Logs de meta-treino (gerados automaticamente)
│
├── outputs/
│   ├── checkpoints/            # Pesos salvos (não versionados)
│   └── results/                # Métricas, plots e tabelas comparativas
│
├── tests/
│   ├── test_data.py
│   ├── test_episode_sampler.py
│   └── test_models.py
│
├── docs/
│   ├── references/             # PDFs dos artigos de FSAR
│   └── notes.md                # Anotações de pesquisa
│
├── README.md
├── pyproject.toml
├── setup.py
├── Makefile
└── .gitignore
```

---

## Instalação

### 1. Clonar este repositório

```bash
git clone <url-deste-repo> Thetis
cd Thetis
```

### 2. Instalar dependências com uv

```bash
uv venv
uv sync
```

> Para executar comandos Python sem ativar manualmente o ambiente virtual, use `uv run <comando>`.

### 3. Clonar o dataset THETIS

```bash
git clone https://github.com/THETIS-dataset/dataset dataset
```

> O dataset contém vídeos pesados (dezenas de GB). A pasta `dataset/` está no `.gitignore` e **não deve ser versionada**.

### 4. Pré-processar dados e extrair modalidades complementares

```bash
make preprocess
# ou diretamente:
uv run python src/data/loader.py --input dataset/ --output data/ --seed 42
uv run python src/features/pose.py         --input data/processed/ --output data/processed/pose/
uv run python src/features/optical_flow.py --input data/processed/ --output data/processed/flow/
uv run python src/features/text.py         --output data/processed/text/
```

Isso gera, entre outros artefatos:

- `data/processed/manifest.csv`: tabela por amostra (sujeito, ação, modalidade, sequência, caminho).
- `data/processed/excluded_clips.json`: clipes descartados da amostragem por
  modalidade (em branco e duplicatas byte a byte). Consumido pelo sampler quando
  `data.exclude_defective: true`, que é o padrão nos 10 configs.
- `data/processed/integrity_report.json`: relatório de integridade e cobertura por modalidade/classe.
  Inclui três checagens de defeito de dados: `key_collisions` (dois arquivos da
  mesma modalidade caindo na mesma chave `(actor, ação, sequência)`),
  `duplicate_files` (clipes byte a byte idênticos — o THETIS preenche repetições
  faltantes copiando outro take: 28 grupos, 56 arquivos) e
  `cross_modality_alignment` (linhas cujas modalidades discordam na contagem de
  frames) e `degenerate_clips` (vídeos totalmente em branco — a segmentação do
  Kinect falha e grava 27 máscaras 100% pretas). Essas exigem `--full-integrity`.
  O `make preprocess` imprime um resumo. Ver
  [`experiments/configs/README.md`](experiments/configs/README.md) para o que cada
  uma significa, o defeito já corrigido no `skeleton_2d` e o impacto medido em
  cada modalidade.
- `data/processed/counts_by_modality_action.csv`: contagens por modalidade e ação.
- `data/processed/pose/`, `data/processed/flow/`, `data/processed/text/`: features das modalidades complementares.

### 5. Construir os splits episódicos

O default da Fase 2 é a partição **6/3/3** das 12 classes com **n_way assimétrico** (5-way em meta_train; 3-way em meta_val/meta_test). Esse é o máximo viável dado o orçamento de 12 classes — ver a seção [Splits](#splits) abaixo.

```bash
make episodes-6-3-3
# ou diretamente:
uv run python src/data/episode_sampler.py \
    --manifest data/processed/manifest.csv \
    --output data/episodes/ \
    --n-way 5 --n-way-val 3 --n-way-test 3 \
    --k-shot 5 --q-query 15 \
    --train-classes 6 --val-classes 3 --test-classes 3 \
    --episodes-per-split 1000 \
    --seed 42
```

Parâmetros do alvo `make episodes` (sobrescreva via linha de comando):

- `N_WAY=5`, `N_WAY_VAL=3`, `N_WAY_TEST=3`, `K_SHOT=5`, `Q_QUERY=15`
- `EPISODES_PER_SPLIT=1000`
- `N_TRAIN=6`, `N_VAL=3`, `N_TEST=3`
- `MANIFEST=data/processed/manifest.csv`, `EPISODES_DIR=data/episodes/`, `SEED=42`

Exemplo de variação 1-shot:

```bash
make episodes-6-3-3 K_SHOT=1
```

Esse comando produz:

- `data/episodes/meta_train/`, `meta_val/`, `meta_test/`: episódios serializados (JSONL).
- `data/episodes/split_metadata.json`: classes em cada partição, n_way por split, seed e parâmetros (N, K, Q).

A divisão de classes entre meta-train/val/test é fixada por configuração para garantir que classes vistas em meta-treino não apareçam em meta-teste.

#### Splits

THETIS tem **12 classes**, o que não comporta 5-way em todos os splits ao mesmo tempo. A escolha desta fase:

| Split | Classes | n_way | Justificativa |
| --- | --- | --- | --- |
| meta_train | 6 | 5 | Maior n_way viável para reportar como headline. |
| meta_val | 3 | 3 | Todas as 3 classes em cada episódio (limite duro pelo orçamento). |
| meta_test | 3 | 3 | Idem. |

Isso registra **n_way_per_split** em `split_metadata.json` e diverge intencionalmente do "5-way puro" da literatura — alternativa seria reduzir tudo para 4-way (4/4/4). A comparação entre métodos continua justa porque todos rodam sob o mesmo protocolo. Tennis7 entra na Fase futura como dataset complementar para reportar 5-way uniforme.

---

## Uso

### Meta-treinar o baseline ProtoNet

```bash
# RGB, 5-way 5-shot (val/test rodam 3-way conforme o split 6/3/3)
make train TRAIN_CONFIG=experiments/configs/protonet_rgb_5w5s.yaml

# ou diretamente:
uv run python src/training/meta_trainer.py \
    --config experiments/configs/protonet_rgb_5w5s.yaml
```

Smoke-test do pipeline (1 epoch × 2 episódios, encoder com peso aleatório — bom para validar configs novas em segundos):

```bash
uv run python src/training/meta_trainer.py \
    --config experiments/configs/protonet_rgb_5w5s.yaml \
    --smoke
```

Cada run grava, **a cada época**:

- `outputs/checkpoints/<run_id>/last.pt` (modelo + optimizer + histórico) e `best.pt` (melhor val_acc)
- `experiments/logs/<run_id>/training.json` (curvas de loss/acc, segundos por época, config + splits)

### Retomar um treino interrompido

`--resume` continua da época seguinte à do `last.pt` — o alvo é `optim.epochs`
da config, então dá para treinar em blocos (20 → 40 → 50 épocas) aumentando
esse número entre as sessões:

```bash
# retoma de outputs/checkpoints/<run_id>/last.pt (começa do zero se não houver)
uv run python src/training/meta_trainer.py \
    --config experiments/configs/protonet_rgb_5w1s.yaml --resume

# ou de um checkpoint específico
uv run python src/training/meta_trainer.py \
    --config experiments/configs/protonet_rgb_5w1s.yaml \
    --resume outputs/checkpoints/<run_id>/last.pt
```

Os episódios são endereçados por índice e o sampler é determinístico por índice,
então a run retomada vê a mesma sequência de episódios de uma run contínua (só o
RNG de augmentation reinicia).

### Treinar no Kaggle

Quem não tem GPU local roda por `notebooks/kaggle_train_protonet.ipynb`, que
clona este repositório, monta a config derivada e treina. Ele serve as cinco
modalidades × {1-shot, 5-shot}: escolha `MODALITY` e `K_SHOT` na seção 1 e o
resto do notebook se ajusta.

Os dados entram como *Dataset* anexado, o manifesto é regenerado a partir da
árvore da modalidade, e o treino usa `--resume`, porque uma sessão do Kaggle
(~12 h) não cobre as 50 épocas.

### Avaliação episódica

```bash
make eval CHECKPOINT=outputs/checkpoints/<run_id>/best.pt

# ou diretamente, escolhendo o JSONL de episódios:
uv run python src/training/eval_episodic.py \
    --checkpoint outputs/checkpoints/<run_id>/best.pt \
    --episodes data/episodes/meta_test/episodes.jsonl
```

A métrica principal é **acurácia média sobre N episódios de teste**, reportada com **intervalo de confiança de 95%** (Student-t), conforme convenção da literatura de FSAR. Os resultados ficam em `outputs/results/<run_id>/`:

- `metrics.json`: mean ± IC 95%, acurácias por episódio, lista de classes.
- `confusion.csv` + `confusion.png`: matriz de confusão por classe canônica.

### Rodar os testes

```bash
uv run pytest tests/
```

### Comandos via Makefile

```bash
make preprocess        # constrói manifest, integrity report e label index
make episodes-6-3-3    # constrói os splits 6/3/3 com n_way assimétrico (5/3/3)
make train             # meta-treina (default: ProtoNet RGB 5w5s)
make eval              # avalia um checkpoint em meta_test
make test              # roda a suíte de testes
make clean             # limpa logs e arquivos temporários
```

---

## Convenções de nomenclatura do THETIS

Cada arquivo de vídeo segue o padrão `{actor}_{action}_{sequence}.avi`.

| Código no arquivo | Ação |
| --- | --- |
| `backhand` | Backhand |
| `backhand2h` | Backhand com duas mãos |
| `bslice` | Backhand slice |
| `foreflat` | Forehand flat |
| `foreopen` | Forehand open stance |
| `fslice` | Forehand slice |
| `serflat` | Serviço flat |
| `serkick` | Serviço kick |
| `serslice` | Serviço slice |
| `smash` | Smash |
| `fvolley` | Forehand volley |
| `bvolley` | Backhand volley |

---

## Configuração de experimentos

Cada experimento é definido por um arquivo `.yaml` em `experiments/configs/`.
Todos seguem o **mesmo schema**, com todas as chaves lidas pelo código escritas
explicitamente — entre dois configs quaisquer só mudam `run_id`, `modalities` e
`episode.k_shot`. A referência completa de cada chave (defaults, efeito em
VRAM/velocidade, limitações conhecidas) está em
[`experiments/configs/README.md`](experiments/configs/README.md).

```yaml
# experiments/configs/protonet_skeleton3d_5w5s.yaml
method: protonet              # protonet | trx | mvp_shot | safsar | vpd | ...
run_id: protonet_skeleton3d_5w5s   # fixa outputs/checkpoints/<run_id> e logs/<run_id>

modalities:                   # a Fase 2 aceita uma modalidade por config
  - skeleton_3d

encoder:
  name: r2plus1d_18
  pretrained: true
  batch_size: 8               # vídeos por forward; principal knob de OOM
  gradient_checkpointing: true

episode:
  n_way: 5
  n_way_val: 3                # split 6/3/3: val/test rodam 3-way
  n_way_test: 3
  k_shot: 5
  q_query: 15
  episodes_per_epoch: 200
  episodes_meta_val: 100
  episodes_meta_test: 1000

optim:
  epochs: 50
  learning_rate: 0.0001
  weight_decay: 0.0
  eval_every: 5
  fp16: true                  # autocast AMP (CUDA)
  stream_query: true          # query em micro-batches + acumulação de gradiente

data:
  manifest_path: data/processed/manifest.csv
  dataset_root: dataset
  train_classes: 6
  val_classes: 3
  test_classes: 3
  frame_count: 16
  resize_size: 128
  spatial_size: 112
  cache_decoded: true         # decode 1x -> RAM (~1,9 GB); remove o gargalo de I/O

seed: 42
```

O baseline ProtoNet cobre as 5 modalidades × {1-shot, 5-shot} — 10 configs, um
por combinação:

| Modalidade | 5-way 1-shot | 5-way 5-shot |
| --- | --- | --- |
| `rgb` | `protonet_rgb_5w1s.yaml` | `protonet_rgb_5w5s.yaml` |
| `depth` | `protonet_depth_5w1s.yaml` | `protonet_depth_5w5s.yaml` |
| `mask` | `protonet_mask_5w1s.yaml` | `protonet_mask_5w5s.yaml` |
| `skeleton_2d` | `protonet_skeleton2d_5w1s.yaml` | `protonet_skeleton2d_5w5s.yaml` |
| `skeleton_3d` | `protonet_skeleton3d_5w1s.yaml` | `protonet_skeleton3d_5w5s.yaml` |

Cenários previstos:

- **5-way 1-shot** e **5-way 5-shot**, sobre as 12 classes do THETIS;
- variantes de modalidade: RGB puro, esqueleto, RGB + pose, RGB + fluxo óptico, RGB + texto (SAFSAR), e combinações;
- subgrupo de robustez a velocidade: episódios em que o suporte vem de iniciantes (p1–p31) e a consulta de especialistas (p32–p55), e vice-versa.

---

## Contribuição esperada

- Comparação sistemática entre famílias de métodos de FSAR aplicadas ao tênis.
- Protocolo de avaliação episódica reproduzível sobre o THETIS, com intenção de extensão ao Tennis7.
- Análise das limitações de cada família frente à variação de velocidade entre jogadores e à granularidade fina entre golpes vizinhos.
- Avaliação do ganho trazido por modalidades complementares (pose 2D, fluxo óptico, texto) em relação a baselines puramente RGB.

---

## Citação

Se este trabalho usar o dataset THETIS, cite:

```bibtex
@inproceedings{gourgari2013thetis,
  title     = {THETIS: Three dimensional tennis shots a human action dataset},
  author    = {Gourgari, S. and Goudelis, G. and Karpouzis, K. and Kollias, S.},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition workshops},
  pages     = {676--681},
  year      = {2013}
}
```
