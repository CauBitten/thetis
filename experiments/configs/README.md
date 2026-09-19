# Configs de experimento

Um `.yaml` por experimento (método × modalidade × N-way × K-shot). Todos os
configs deste diretório seguem **o mesmo schema**, com todas as chaves lidas
pelo código escritas explicitamente — nada fica no default implícito. Isso é o
que garante que a diferença de resultado entre duas runs venha da modalidade ou
do K, e não do hardware onde rodou.

Entre dois configs quaisquer, só três coisas mudam: `run_id`, `modalities` e
`episode.k_shot`.

## Convenção de nomes

```text
protonet_<modalidade>_5w<K>s.yaml      →   run_id = protonet_<modalidade>_5w<K>s
```

A modalidade no nome do arquivo vai sem underscore (`skeleton2d`), enquanto o
valor em `modalities:` usa o identificador do código (`skeleton_2d`).

O `run_id` fixo define `outputs/checkpoints/<run_id>/` e
`experiments/logs/<run_id>/`. É o que faz `--resume` funcionar pela CLI: sem
ele, o trainer gera `protonet_<mod>_<k>s_<timestamp>` e cada invocação cria um
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
| `method` | — (obrigatória) | Só `protonet` é implementado (`meta_trainer.py:75`). |
| `run_id` | `protonet_<mod>_<k>s_<timestamp>` | Nome dos diretórios de checkpoint e log. Ver acima. |
| `modalities` | — (obrigatória) | Lista de 1 elemento; a Fase 2 aceita uma modalidade por config. Válidos: `rgb`, `depth`, `mask`, `skeleton_2d`, `skeleton_3d`. |
| `seed` | — (obrigatória) | Semente de NumPy/Torch/CUDA e do sampler de episódios. |
| `output_root` | `outputs` | Raiz dos checkpoints. Os notebooks sobrescrevem. |
| `log_root` | `experiments/logs` | Raiz dos logs. Os notebooks sobrescrevem. |

### `encoder`

| Chave | Default | Efeito |
| --- | --- | --- |
| `name` | `r2plus1d_18` | Backbone de vídeo (torchvision). |
| `pretrained` | `true` | Pesos Kinetics-400. Ignorado em `--smoke` (usa peso aleatório). |
| `batch_size` | auto por VRAM | Quantos vídeos passam pelo encoder por forward (encoding em chunks). Knob de OOM **e hiperparâmetro de treino** — ver abaixo. |
| `gradient_checkpointing` | auto: só liga em GPU ≤6 GB | Recomputa ativações no backward: ~70% menos VRAM de ativação, ~25-30% mais lento por step. Não altera o resultado. |

**`encoder.batch_size` não é um knob puramente de memória.** O `ProtoNet._encode`
fatia o lote em chunks desse tamanho (`src/models/protonet.py:62-65`), e o
R(2+1)D-18 tem **37 camadas `BatchNorm3d`** sem congelamento. Em `model.train()`
cada chunk é normalizado pelas **próprias estatísticas**, então o tamanho do
chunk muda as ativações, os gradientes e os `running_mean/var` que depois são
usados na avaliação. Medido com o mesmo lote e a mesma seed:

| Modo | Diferença máxima absoluta entre os embeddings de `bs=8` e `bs=16` |
| --- | --- |
| `train()` | **1.305** |
| `eval()` | 0.000 |

Ou seja: **duas runs com `batch_size` diferente não são comparáveis**, mesmo com
tudo o mais idêntico. Por isso os configs fixam `16` explicitamente e **o
notebook do Kaggle não sobrescreve mais esse valor** — ele faz parte do
protocolo experimental, não da configuração da máquina.

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

Se precisar de folga de memória, mexa antes em `gradient_checkpointing` e
`optim.stream_query`: os dois reduzem o pico de VRAM **sem alterar o resultado**.

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
| `stream_query` | `true` em CUDA | Processa o conjunto de query em micro-batches com acumulação de gradiente; limita o pico de VRAM. |
| `cuda_memory_fraction` | `0.92` | Teto da fração de VRAM por processo. Faz o OOM falhar rápido no limite real em vez de vazar para a RAM compartilhada (relevante no Windows). Não está escrito nos configs; ajuste só se precisar. |

### `data`

| Chave | Default | Efeito |
| --- | --- | --- |
| `manifest_path` | — (obrigatória) | `data/processed/manifest.csv`, gerado por `make preprocess`. |
| `dataset_root` | — (obrigatória) | Raiz dos `VIDEO_*` do THETIS. |
| `train_classes` / `val_classes` / `test_classes` | `6` / `3` / `3` | Partição das 12 classes; disjunta por construção. |
| `frame_count` | `16` | Frames por clipe após o crop temporal. O dataset decodifica `2×` isso e o `RandomTemporalCrop` reduz. |
| `resize_size` | `128` | Lado menor após resize, antes do crop. |
| `spatial_size` | `112` | Lado do crop final que entra no encoder. |
| `cache_decoded` | `true` | Decodifica cada clipe uma vez, redimensiona para `resize_size` e serve da RAM. |
| `exclude_defective` | `true` | Descarta do pool os clipes listados em `data/processed/excluded_clips.json` (em branco e duplicatas). Ver "Integridade do dataset". |

**`data.cache_decoded`.** A amostragem episódica reaproveita o mesmo pool de
~990 clipes de treino milhares de vezes, então o decode serial de vídeo — não a
GPU — é o gargalo real. Cachear custa ~1,9 GB de RAM a 128² (contra ~29 MB por
clipe em 480×640 nativo) e não muda o resultado: o resize do cache é o mesmo
`ResizeVideo` do transform. Desligue só se estiver limitado de RAM.

## Limitações conhecidas

Registradas aqui porque afetam a leitura dos resultados, mas são comportamento
atual do código e não configuráveis:

- **Normalização Kinetics em todas as modalidades.** As estatísticas de
  normalização são as do Kinetics-400 RGB e são aplicadas a qualquer modalidade
  (`src/models/encoders.py:26,57-59`), inclusive `depth`, `mask` e os vídeos de
  esqueleto, que não são RGB natural. O mesmo vale para `encoder.pretrained`.
- **Augmentation fixa no código.** `build_train_transform`
  (`src/training/meta_trainer.py:215-226`) não lê parâmetros do config:
  `RandomTemporalCrop → ResizeVideo → RandomSpatialCrop → HorizontalFlip(p=0.5)
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
