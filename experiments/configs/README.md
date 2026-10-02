# Configs de experimento

Um `.yaml` por experimento (método × modalidade × N-way × K-shot). Todos os
configs deste diretório seguem **o mesmo schema**, com todas as chaves lidas
pelo código escritas explicitamente — nada fica no default implícito. Isso é o
que garante que a diferença de resultado entre duas runs venha da modalidade ou
do K, e não do hardware onde rodou.

Entre dois configs do mesmo método, só três coisas mudam: `run_id`,
`modalities` e `episode.k_shot`. Entre métodos, muda só o que é do método
(`method`, `encoder.name`, `model` e a amostragem temporal); splits, episódios,
seed, otimizador e avaliação são os mesmos — ver "TRX" abaixo.

## Convenção de nomes

```text
<método>_<modalidade>_5w<K>s.yaml      →   run_id = <método>_<modalidade>_5w<K>s
```

A modalidade no nome do arquivo vai sem underscore (`skeleton2d`), enquanto o
valor em `modalities:` usa o identificador do código (`skeleton_2d`).

O `run_id` fixo define `outputs/checkpoints/<run_id>/` e
`experiments/logs/<run_id>/`. É o que faz `--resume` funcionar pela CLI: sem
ele, o trainer gera `<method>_<mod>_<k>s_<timestamp>` e cada invocação cria um
diretório novo, onde o `last.pt` nunca é encontrado. Como contrapartida,
**re-rodar o mesmo config sobrescreve a run anterior** — para variar
hiperparâmetros, copie o arquivo com outro nome (e outro `run_id`). O notebook
do Kaggle sobrescreve `cfg["run_id"]` depois de carregar, então não é afetado.

## Configs disponíveis

| Modalidade | 5-way 1-shot | 5-way 5-shot |
| --- | --- | --- |
| `rgb` | `protonet_rgb_5w1s.yaml` | `protonet_rgb_5w5s.yaml` |
| `depth` | `protonet_depth_5w1s.yaml` | `protonet_depth_5w5s.yaml` |
| `mask` | `protonet_mask_5w1s.yaml` | `protonet_mask_5w5s.yaml` |
| `skeleton_2d` | `protonet_skeleton2d_5w1s.yaml` | `protonet_skeleton2d_5w5s.yaml` |
| `skeleton_3d` | `protonet_skeleton3d_5w1s.yaml` | `protonet_skeleton3d_5w5s.yaml` |

Os 10 do TRX seguem o mesmo padrão, com prefixo `trx_` (`trx_rgb_5w1s.yaml`,
…, `trx_skeleton3d_5w5s.yaml`).

Arquivos com prefixo `_` (`_kaggle_active.yaml`) são derivados gerados pelo
notebook — não edite à mão.

## Cobertura de dados por modalidade

De `data/processed/counts_by_modality_action.csv` (12 classes, 1980 linhas no
manifesto):

| Modalidade | Clipes/classe | Total | Cobertura |
| --- | --- | --- | --- |
| `rgb` | 165 | 1980 | 100% |
| `depth` | 165 | 1980 | 100% |
| `mask` | 165 | 1980 | 100% |
| `skeleton_2d` | 93–110 | 1217 | ~61% |
| `skeleton_3d` | 93–110 | 1217 | ~61% |

O sampler descarta linhas onde `path_<modality>` está vazio
(`src/data/episode_sampler.py:131-137`), então as modalidades de esqueleto
amostram de um pool menor. Ainda assim há folga: 5-way 5-shot precisa de
`k_shot + q_query` = 20 clipes por classe, contra 93 no pior caso.

### Integridade do dataset

Quatro checagens no `integrity_report.json`. Só a primeira roda sempre; as outras
exigem `--full-integrity`, que é o que o `make preprocess` faz (~25 s no THETIS
completo):

```bash
make preprocess
# equivale a:
uv run python src/data/loader.py --input dataset --output data --seed 42 --full-integrity
```

#### `key_collisions` — dois arquivos caindo na mesma chave

Dois arquivos da mesma modalidade que parseiam para a mesma chave
`(actor, ação, sequência)`. O último vence, então o outro sumiria do manifesto
sem aviso. Roda sempre; a CLI imprime `WARNING` quando acha algo.

> **Caso corrigido.** As sequências 2 e 3 de `p19`/`backhand_volley` chegaram
> nomeadas `p19_bvolley_skelet2D_s3 (1).avi` e `(2).avi`. O sufixo ` (N)` é
> removido na normalização, as duas colidiam em `(p19, backhand_volley, 3)`, e o
> resultado era: a linha `s3` pareava RGB da sequência 3 com esqueleto da
> sequência **2**, a linha `s2` ficava sem `skeleton_2d`, e o arquivo da
> sequência 3 era descartado. A contagem de frames identificou cada arquivo —
> `(1)` tem 82 frames (= RGB s3 e Skelet3D s3), `(2)` tem 83 (= RGB s2 e
> Skelet3D s2) — e ambos foram renomeados para `_s3.avi` e `_s2.avi`.
> `skeleton_2d` foi de 1216 para **1217**, que é exatamente a contagem
> documentada no `dataset/README.md` do THETIS. `key_collisions` agora vem vazio.
>
> Como `dataset/` não é versionado, **um download novo traz os nomes originais de
> volta.** O `make preprocess` avisa, e o conserto é repetir o rename.

#### `duplicate_files` — clipes byte a byte idênticos

O THETIS preenche repetições que não existem **copiando outro take**. São
**28 grupos, 56 arquivos**, confirmados por MD5 (nenhum processamento envolvido —
os arquivos são idênticos no disco):

| Padrão | Onde |
| --- | --- |
| `s3` inteira é cópia da `s2` (todas as modalidades) | `p47`/`backhand`, `p55`/`backhand_slice` |
| `s2` inteira é cópia da `s1` | `p50`/`forehand_volley` |
| **`depth`/`rgb` vêm da `s2` mas `mask` vem da `s1`** | `p55` × `backhand2hands`, `flat_service`, `forehand_flat`, `forehand_openstands`, `forehand_slice` |
| **`rgb` da `s2` é cópia da `s3`**, resto é genuíno | `p13`/`forehand_flat` |
| **duas `mask` 100% pretas** (idênticas por serem ambas vazias) | `p1_backhand_mask_s1` == `p11_serslice_mask_s1` |

As três últimas linhas são **inconsistências internas**: a linha mistura
modalidades de takes diferentes. Confirmado independentemente por um teste de
sincronia `depth`↔`mask` (a silhueta cai sobre uma região coerente de
profundidade só se as duas forem o mesmo take, com deslocamento temporal como
controle): em 24 linhas normais o lag 0 vence em **24/24**; nas 5 linhas do `p55`
com `depth`/`mask` de comprimentos diferentes, o lag 0 **perde** em 5/5.

Impacto no pool de treino de cada modalidade:

| Modalidade | Clipes em grupo duplicado | % do pool |
| --- | --- | --- |
| `rgb` | 18 | 0,91% |
| `depth` | 16 | 0,81% |
| `mask` | 18 | 0,91% |
| `skeleton_2d` | 2 | 0,16% |
| `skeleton_3d` | 2 | 0,16% |

Efeito real: um episódio pode sortear o mesmo clipe para suporte **e** consulta,
o que deixa a acurácia otimista. O par `p1_backhand_mask_s1` ==
`p11_serslice_mask_s1` **não** é contaminação entre classes, como parecia à
primeira vista: os dois vídeos são 100% pretos, e batem em MD5 justamente por
ambos serem vazios. Os arquivos seguem como o THETIS os distribui — o que muda é
que agora eles são filtrados na amostragem (ver abaixo).

#### `cross_modality_alignment` — modalidades discordando na contagem de frames

**10 linhas de 1980**, com diferenças de 2 a 32 frames. Cruzando com
`duplicate_files`, elas se separam em duas causas:

- **5 são defeito de cópia** (as do `p55` acima): `depth` e `mask` vêm de takes
  diferentes, e é por isso que os comprimentos não batem.
- **5 são diferença de janela de gravação/trim** entre pipelines — `depth` e
  `mask` seguem sincronizados (lag 0 vence), só o comprimento difere. Ex.:
  `p24_forehand_flat_s1` (`rgb`=108, `depth`/`mask`=140, ambos sincronizados) e
  `p21_kick_service_s3` (`skeleton_2d`=54 contra 68 nas demais). Isso é
  propriedade do dataset, não defeito.

Um teste anterior por correlação de perfil de movimento foi **descartado**: no
controle, 80,7% dos pares comprovadamente de takes diferentes passavam pelo
limiar, ou seja, o teste não tinha poder discriminativo entre modalidades de
aparência tão distinta.

#### `degenerate_clips` — clipes sem sinal nenhum

A segmentação de jogador do Kinect falha por completo em algumas gravações e
grava um vídeo `mask` inteiramente preto. São arquivos válidos, com rótulo
válido, que passam em todas as outras checagens — e não ensinam nada.

**27 máscaras (1,4%) são 100% pretas.** A cauda é mais longa: 5,4% têm mais de
metade dos frames sem silhueta e 22% têm mais de um quarto. A detecção é
genérica (nenhum frame acima do ruído do sensor), então pega um clipe em branco
em qualquer modalidade, não só `mask`.

#### Filtro de amostragem (`data.exclude_defective`)

O `make preprocess --full-integrity` grava
`data/processed/excluded_clips.json`, e o sampler descarta esses `sample_id` do
pool da modalidade. Todos os 10 configs trazem `data.exclude_defective: true`.

| Modalidade | Clipes excluídos | Motivo |
| --- | --- | --- |
| `rgb` | 9 | duplicatas |
| `depth` | 8 | duplicatas |
| `mask` | 35 | 27 em branco + 8 duplicatas |
| `skeleton_2d` | 1 | duplicata |
| `skeleton_3d` | 1 | duplicata |

Duas regras, só:

1. **em branco** — sem sinal para aprender;
2. **duplicata byte a byte** — mantém-se um representante por grupo (o menor
   `sample_id`, para a escolha ser estável entre execuções) e descartam-se as
   cópias, o que elimina o vazamento suporte↔consulta.

As linhas cujas modalidades vêm de takes diferentes são **registradas mas não
excluídas** (`rows_sharing_clips_across_takes`): cada clipe continua sendo um
exemplo válido da sua própria modalidade, e os configs monomodais nunca os
pareiam. A lista existe para o trabalho multimodal, onde essas linhas precisam
sair.

O filtro vale igualmente para treino e para o `meta_test`, então a avaliação não
é feita sobre clipes em branco nem sobre material duplicado. Se
`excluded_clips.json` não existir (árvore que nunca rodou `--full-integrity`), o
treino segue sem filtro e avisa no log.

#### O que isso significa para os experimentos

**Nenhuma dessas checagens acusa erro no código deste repositório.** O loader
pareia estritamente por nome de arquivo; as duplicatas são idênticas em MD5 no
disco, como o THETIS as distribui.

Para os configs atuais (uma modalidade por vez), os defeitos afetam ≤1,8% do pool
e agora são filtrados automaticamente. Para o trabalho multimodal previsto
(RGB + pose, SAFSAR), as linhas em `rows_sharing_clips_across_takes` também
precisam sair — o filtro atual não as remove, de propósito.

**Runs anteriores a esta mudança não têm o filtro** e portanto incluem os clipes
duplicados/em branco. Para `mask` isso é 1,8% do pool; para as demais, <1%.

## Referência das chaves

Defaults são os do código quando a chave é omitida — nos configs deste
diretório nenhuma delas é omitida.

### Topo

| Chave | Default | Efeito |
| --- | --- | --- |
| `method` | — (obrigatória) | Cabeça few-shot, registrada em `METHODS` (`src/models/factory.py`). Hoje: `protonet`, `trx`. |
| `model` | `{}` | Hiperparâmetros do método, repassados como kwargs ao construtor da cabeça. O ProtoNet não tem nenhum; os do TRX estão em "TRX" abaixo. |
| `run_id` | `<method>_<mod>_<k>s_<timestamp>` | Nome dos diretórios de checkpoint e log. Ver acima. |
| `modalities` | — (obrigatória) | Lista de 1 elemento; a Fase 2 aceita uma modalidade por config. Válidos: `rgb`, `depth`, `mask`, `skeleton_2d`, `skeleton_3d`. |
| `seed` | — (obrigatória) | Semente de NumPy/Torch/CUDA e do sampler de episódios. |
| `output_root` | `outputs` | Raiz dos checkpoints. Os notebooks sobrescrevem. |
| `log_root` | `experiments/logs` | Raiz dos logs. Os notebooks sobrescrevem. |

### `encoder`

| Chave | Default | Efeito |
| --- | --- | --- |
| `name` | `r2plus1d_18` | Backbone (torchvision). **Vídeo**, devolve `(B, D)`: `r2plus1d_18`, `r3d_18`. **Por frame**, devolve `(B, T, D)`: `resnet18`, `resnet34` (D=512), `resnet50` (D=2048, o do TRX). Métodos que comparam frames (TRX) exigem um encoder por frame; o ProtoNet aceita os dois (faz a média no tempo). |
| `pretrained` | `true` | Pesos Kinetics-400 (vídeo) ou ImageNet `IMAGENET1K_V1` (por frame). Ignorado em `--smoke` (usa peso aleatório). |
| `batch_size` | auto por VRAM | Quantos vídeos passam pelo encoder por forward (encoding em chunks). Knob de OOM **e hiperparâmetro de treino** — ver abaixo. |
| `gradient_checkpointing` | auto: só liga em GPU ≤6 GB | Recomputa ativações no backward: ~70% menos VRAM de ativação, ~25-30% mais lento por step. **Altera o resultado** (estatísticas do BatchNorm) — faz parte do protocolo, ver abaixo. |

**`encoder.batch_size` não é um knob puramente de memória.** O `EpisodicModel._encode`
fatia o lote em chunks desse tamanho (`src/models/base.py:74-75`), e o
R(2+1)D-18 tem **37 camadas `BatchNorm3d`** sem congelamento. Em `model.train()`
cada chunk é normalizado pelas **próprias estatísticas**, então o tamanho do
chunk muda as ativações, os gradientes e os `running_mean/var` que depois são
usados na avaliação. Medido com o mesmo lote e a mesma seed:

| Modo | Diferença máxima absoluta entre os embeddings de `bs=8` e `bs=16` |
| --- | --- |
| `train()` | **1.305** |
| `eval()` | 0.000 |

Ou seja: **duas runs com `batch_size` diferente não são comparáveis**, mesmo com
tudo o mais idêntico. Nos encoders por frame vale o mesmo: as `BatchNorm2d` da
ResNet normalizam os `batch_size × frame_count` frames de cada chunk. Por isso
os configs fixam `16` explicitamente e **o notebook do Kaggle não sobrescreve
esse valor** — ele faz parte do protocolo experimental, não da configuração da
máquina.

`16` foi escolhido por ser o valor que roda em T4/P100/L4 (as GPUs de fato
usadas) com `gradient_checkpointing` e `stream_query` ligados. Numa GPU menor que
~12 GB isso pode dar OOM; nesse caso **não baixe só um config** — ou baixa em
todos e re-roda a comparação inteira, ou treina numa GPU maior.

Para referência, o auto-detect (usado só quando a chave é omitida — nenhum config
daqui omite) escolheria:

| VRAM | `batch_size` auto |
| --- | --- |
| ≤ 6 GB | 4 |
| ≤ 10 GB | 8 |
| ≤ 16 GB | 16 |
| > 16 GB | 32 |

**`encoder.gradient_checkpointing` e `optim.stream_query` também fazem parte do
protocolo.** Os dois reduzem o pico de VRAM, mas, como os encoders têm
BatchNorm sem congelamento, nenhum deles é neutro:

- **`gradient_checkpointing`**: o backward re-executa o forward de cada estágio
  em modo treino, e cada BatchNorm atualiza `running_mean/var` **duas vezes por
  passo** (`num_batches_tracked` sobe 2 por step). As ativações e os gradientes
  do treino não mudam, mas as estatísticas que a avaliação usa, sim.
- **`stream_query`**: o suporte passa pelo encoder num lote só e a query em
  micro-batches de `batch_size`; desligado, suporte e query são concatenados e
  fatiados em chunks mistos. Muda a composição dos lotes do BatchNorm e, com
  ela, as ativações, a loss e os gradientes. A equivalência exata do streaming
  testada em `tests/test_training.py` só vale para encoders sem BatchNorm.

Todos os configs fixam os dois em `true`, e as runs do ProtoNet foram feitas
assim. Se faltar VRAM, **não desligue nenhum dos dois nem baixe `batch_size`
num config só**: treine numa GPU maior, ou mude em todos e re-rode a comparação.

### Histórico de runs anteriores a esta padronização

Runs gravadas antes de o `batch_size` virar parte do protocolo usaram valores
diferentes e **não são comparáveis entre si** nem com as novas:

| Run | `batch_size` | Observação |
| --- | --- | --- |
| `protonet_rgb_5w5s_colab` | 32 | 100 épocas, best_val=0.916. Refazer para entrar na comparação. |
| `protonet_rgb_5w1s_kaggle` | 16 | 23/50 épocas, best_val=0.796. Compatível com o protocolo atual. |

### `episode`

| Chave | Default | Efeito |
| --- | --- | --- |
| `n_way` | — (obrigatória) | Classes por episódio no meta-train. |
| `n_way_val` | `= n_way` | Classes por episódio no meta-val. |
| `n_way_test` | `= n_way` | Classes por episódio no meta-test. |
| `k_shot` | — (obrigatória) | Exemplos de suporte por classe. |
| `q_query` | — (obrigatória) | Exemplos de consulta por classe. |
| `episodes_per_epoch` | `200` | Episódios de treino por época. |
| `episodes_meta_val` | `100` | Episódios por rodada de validação. |
| `episodes_meta_test` | `1000` | Episódios na avaliação final (`eval_episodic.py`). |

O `5/3/3` (`n_way` / `n_way_val` / `n_way_test`) vem do orçamento de 12 classes
com partição 6/3/3 — ver a seção "Splits" do README raiz.

### `optim`

| Chave | Default | Efeito |
| --- | --- | --- |
| `epochs` | `50` | Alvo de épocas. `--resume` continua até esse número, então dá para treinar em blocos aumentando o valor entre sessões. |
| `learning_rate` | `1e-4` | LR do Adam. |
| `weight_decay` | `0.0` | Weight decay do Adam. |
| `eval_every` | `5` | Intervalo (em épocas) da validação episódica. |
| `fp16` | `true` em CUDA | Autocast AMP; ~metade da VRAM de ativação. Ignorado na CPU. |
| `stream_query` | `true` em CUDA | Processa o conjunto de query em micro-batches com acumulação de gradiente; limita o pico de VRAM. Muda a composição dos lotes do BatchNorm, então faz parte do protocolo (ver `encoder`). |
| `cuda_memory_fraction` | `0.92` | Teto da fração de VRAM por processo. Faz o OOM falhar rápido no limite real em vez de vazar para a RAM compartilhada (relevante no Windows). Não está escrito nos configs; ajuste só se precisar. |

### `data`

| Chave | Default | Efeito |
| --- | --- | --- |
| `manifest_path` | — (obrigatória) | `data/processed/manifest.csv`, gerado por `make preprocess`. |
| `dataset_root` | — (obrigatória) | Raiz dos `VIDEO_*` do THETIS. |
| `train_classes` / `val_classes` / `test_classes` | `6` / `3` / `3` | Partição das 12 classes; disjunta por construção. |
| `frame_count` | `16` | Frames por clipe que entram no encoder. No treino o dataset decodifica `temporal_oversample×` isso e o passo temporal reduz; na avaliação decodifica exatamente `frame_count`, uniforme no clipe inteiro. |
| `temporal_sampling` | `crop` | Passo temporal do treino: `crop` (`RandomTemporalCrop`, janela contígua) ou `segment` (`RandomSegmentSample`, um frame aleatório por segmento, estilo TSN). Não afeta a avaliação. Ver o parágrafo `data.temporal_sampling` abaixo. |
| `temporal_oversample` | `2` | Fator de sobre-decodificação do treino (`frame_count × fator` frames). Com `segment`, define quantos frames candidatos cada segmento tem. |
| `resize_size` | `128` | Lado menor após resize, antes do crop. |
| `spatial_size` | `112` | Lado do crop final que entra no encoder. |
| `cache_decoded` | `true` | Decodifica cada clipe uma vez, redimensiona para `resize_size` e serve da RAM. |
| `exclude_defective` | `true` | Descarta do pool os clipes listados em `data/processed/excluded_clips.json` (em branco e duplicatas). Ver "Integridade do dataset". |

**`data.cache_decoded`.** A amostragem episódica reaproveita o mesmo pool de
~990 clipes de treino milhares de vezes, então o decode serial de vídeo — não a
GPU — é o gargalo real. Cachear custa ~1,9 GB de RAM a 128² (contra ~29 MB por
clipe em 480×640 nativo) e não muda o resultado: o resize do cache é o mesmo
`ResizeVideo` do transform. Desligue só se estiver limitado de RAM.

**`data.temporal_sampling`.** Com `crop` (o default e o que os 10 configs do
ProtoNet usaram), o treino decodifica `2 × frame_count` frames espalhados pelo
clipe e recorta uma janela contígua de `frame_count`: o modelo vê metade do golpe,
em meia velocidade em relação ao que vê na avaliação, que cobre o clipe inteiro.
`segment` divide o clipe em `frame_count` segmentos e sorteia um frame em cada,
então o treino sempre cobre o golpe inteiro, como no TSN e no TRX. A avaliação é
a mesma nos dois modos (sem passo temporal; `frame_count` frames uniformes no
clipe inteiro), então, com a mesma `seed`, os episódios de meta-teste e a regra de
amostragem dos frames são os mesmos, e os números de teste continuam comparáveis
entre métodos treinados com `crop` e com `segment`. O que muda é só a
augmentation temporal do treino — registre o modo usado ao comparar resultados.
Para `segment`, use `temporal_oversample: 4` (com `frame_count: 8` são 32 frames
decodificados, o mesmo custo de cache do ProtoNet com 16 × 2).

## TRX

Implementação em `src/models/trx.py`, portada do código dos autores
(github.com/tobyperrett/trx) e conferida contra ele em
`tests/test_trx.py::test_trx_matches_reference_implementation`.

### `model` (chaves do TRX)

| Chave | Default | Efeito |
| --- | --- | --- |
| `temporal_set_sizes` | `[2, 3]` | Cardinalidades Ω das tuplas de frames; um ramo por valor, logits na média. Com `frame_count: 8` são `C(8,2) + C(8,3) = 28 + 56` tuplas. |
| `d_model` | `1152` | Tamanho das chaves e valores (`trans_linear_out_dim` do artigo). |
| `dropout` | `0.1` | Dropout após o positional encoding. |
| `pe_scale` | `0.1` | Amplitude do positional encoding senoidal. |

O TRX exige um encoder por frame (`encoder.name` = `resnet18/34/50`); com um
encoder de vídeo, o `build_model` falha no início da run.

### O que difere do ProtoNet nos configs

| Chave | ProtoNet | TRX | Por quê |
| --- | --- | --- | --- |
| `encoder.name` | `r2plus1d_18` (Kinetics) | `resnet50` (ImageNet, por frame) | Backbone do artigo; o TRX precisa do eixo temporal. |
| `data.frame_count` | `16` | `8` | Como no artigo; com 16 seriam 680 tuplas e a atenção cresceria ~70×. |
| `data.temporal_sampling` | `crop` | `segment` | Como no artigo. Não afeta a avaliação (ver `data.temporal_sampling`). |
| `data.temporal_oversample` | `2` | `4` | 32 frames decodificados, o mesmo custo de cache. |

### O que difere do artigo

Para manter o protocolo comum a todos os métodos, os configs do TRX **não**
reproduzem o treino do artigo em três pontos:

- **Resolução 112² em vez de 224²** — a mesma entrada do ProtoNet, e ~4× mais
  barato. A ResNet-50 da ImageNet termina num mapa 4×4 em vez de 7×7.
- **Adam (lr 1e-4), um episódio por passo** — o código oficial usa SGD
  (lr 1e-3) acumulando o gradiente de 16 episódios.
- **15 consultas por classe e 3-way em val/test** — o split 6/3/3 do THETIS.

Como o backbone também muda (ResNet-50 ImageNet contra R(2+1)D-18 Kinetics), a
diferença TRX × ProtoNet mistura o efeito da cabeça com o do backbone. Para
isolar a cabeça, dá para rodar o ProtoNet sobre o mesmo encoder: um config com
`method: protonet` e o `encoder`/`data` do TRX (o ProtoNet faz a média das
features por frame — é o baseline do próprio artigo do TRX).

## Limitações conhecidas

Registradas aqui porque afetam a leitura dos resultados, mas são comportamento
atual do código e não configuráveis:

- **Normalização RGB em todas as modalidades.** As estatísticas de
  normalização são as do Kinetics-400 RGB (encoders de vídeo) ou da ImageNet
  (encoders por frame) e são aplicadas a qualquer modalidade
  (`src/models/encoders.py:29-33,69-71`), inclusive `depth`, `mask` e os vídeos de
  esqueleto, que não são RGB natural. O mesmo vale para `encoder.pretrained`.
- **Augmentation fixa no código.** Fora o passo temporal
  (`data.temporal_sampling`), `build_train_transform`
  (`src/training/meta_trainer.py:228-250`) não lê parâmetros do config:
  `<passo temporal> → ResizeVideo → RandomSpatialCrop → HorizontalFlip(p=0.5)
  → ColorJitter(0.2/0.2/0.2)`. Consequências: `ColorJitter` é praticamente
  inócuo em `mask` (silhueta binária), e `HorizontalFlip` inverte a lateralidade
  do golpe em todas as modalidades.
- **Esqueletos são vídeos de visualização.** As modalidades `skeleton_2d` e
  `skeleton_3d` do THETIS são o esqueleto renderizado sobre fundo preto, não
  coordenadas de junta — ver `docs/notes.md`. Por isso passam pelo mesmo encoder
  de vídeo das demais.

## Uso

```bash
# treinar
make train TRAIN_CONFIG=experiments/configs/protonet_depth_5w5s.yaml

# validar um config novo em segundos (CPU, peso aleatório, 1 época × 1 episódio)
uv run python src/training/meta_trainer.py \
    --config experiments/configs/protonet_depth_5w5s.yaml --smoke

# retomar de outputs/checkpoints/<run_id>/last.pt
uv run python src/training/meta_trainer.py \
    --config experiments/configs/protonet_depth_5w5s.yaml --resume
```

O `--smoke` prefixa `smoke_` no `run_id`, então não sobrescreve a run real.
