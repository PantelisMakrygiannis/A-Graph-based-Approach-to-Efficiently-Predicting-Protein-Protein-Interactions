# A Graph-based Approach to Efficiently Predicting Protein-Protein Interactions

This repository contains the official implementation of the framework described in the paper *"A Graph-based Approach to Efficiently Predicting Protein-Protein Interactions"* (IJAIT, under review) by Pantelis Makrygiannis, Nikitas-Rigas Kalogeropoulos, Agorakis Bompotas, and Christos Makris (Department of Computer Engineering and Informatics, University of Patras).

The framework predicts protein-protein interactions (PPIs) by combining **Gene Ontology (GO) semantic embeddings** with **Graph Neural Networks (GNNs)**. Unlike sequence-based methods that treat GO annotations as unstructured text, our approach explicitly models the protein interactome as a graph and propagates semantic information across high-confidence interaction edges. We evaluate three GNN encoders, namely GCN, GraphSAGE, and GATv2, on benchmarks derived from STRING v11.0 and STRING v11.5 for *Homo sapiens* and *Saccharomyces cerevisiae*.

## Highlights

Our GATv2 model achieves an AUROC of **0.977** on human and **0.974** on yeast on the standard STRING v11.0 benchmark, surpassing the TransformerGO baseline of 0.974 and 0.961 respectively. Even our simpler GCN encoder outperforms the transformer-based baseline on both organisms, demonstrating that explicitly modeling network topology is at least as important as semantic content for accurate PPI prediction.

## Repository Structure

```
.
|-- datasets/
|   |-- v11.0/                                  # STRING v11.0 raw data and processed splits
|   |   |-- 4932.protein.aliases.v11.0.txt.gz   # Yeast aliases
|   |   |-- 4932.protein.links.v11.0.txt.gz     # Yeast PPI links
|   |   |-- 9606.protein.aliases.v11.0.txt.gz   # Human aliases
|   |   |-- 9606.protein.links.v11.0.txt.gz     # Human PPI links
|   |   |-- goa_human.gaf.gz                    # GO annotation file (human)
|   |   |-- sgd.gaf.gz                          # GO annotation file (yeast)
|   |   |-- go-terms/
|   |   |   |-- go-basic.obo                    # GO ontology (OBO format)
|   |   |   |-- go-basic.obo.csv                # Parsed OBO -> CSV
|   |   |   |-- go_id_dict                      # GO term -> integer index (pickle)
|   |   |   |-- go_namespace_dict               # GO term -> namespace (pickle)
|   |   |   |-- graph/go-terms.edgelist         # GO DAG edge list (is_a / part_of)
|   |   |   `-- embeddings/go-terms.emb         # node2vec GO term embeddings (d=64)
|   |   `-- interaction-datasets/
|   |       |-- PPIv11.0.ipynb                  # Builds the fixed train/valid/test no-mirror splits
|   |       |-- train/, valid/, test/           # Pre-split positive and negative interactions
|   |
|   `-- v11.5/                                  # STRING v11.5 (TransformerGO-derived datasets)
|       |-- 4932.protein.aliases.v11.5.txt.gz
|       |-- 4932.protein.links.v11.5.txt.gz
|       |-- 9606.protein.aliases.v11.5.txt.gz
|       |-- 9606.protein.links.v11.5.txt.gz
|       |-- goa_human.gaf.gz
|       |-- sgd.gaf.gz
|       |-- go-terms/                           # Same layout as v11.0
|       `-- interaction-datasets/
|           |-- PPIv11.5.ipynb                  # Builds positive / negative interactions
|           |-- 4932.protein.positive.v11.5.txt
|           |-- 4932.protein.negative.v11.5.txt
|           |-- 9606.protein.positive.v11.5.txt
|           `-- 9606.protein.negative.v11.5.txt
|
|-- term-encoding-module/                       # GO -> node2vec pipeline
|   |-- OBOParsing.ipynb                        # Parses go-basic.obo into a CSV
|   |-- node2vecTerms.ipynb                     # Builds the GO edge list and helper dictionaries
|   `-- node2vec-embeddings.py                  # Trains node2vec embeddings on the GO graph
|
`-- training-testing/
    `-- GNN-Training.ipynb                      # Data preprocessing + GCN / GATv2 / GraphSAGE training
```

## Framework Overview

The pipeline is organized into four stages.

**Stage 1 – GO ontology parsing.** `term-encoding-module/OBOParsing.ipynb` reads `go-basic.obo`, removes obsolete terms, and emits a tabular CSV. `term-encoding-module/node2vecTerms.ipynb` then keeps only the `is_a` and `part_of` relations, assigns a unique integer index to each GO term, and writes (i) the GO DAG as an edge list (`go-terms.edgelist`), (ii) a `GO term -> index` dictionary (`go_id_dict`), and (iii) a `GO term -> namespace` dictionary (`go_namespace_dict`, with namespaces BP, MF, CC).

**Stage 2 – GO term embeddings.** `term-encoding-module/node2vec-embeddings.py` runs node2vec (biased random walks + skip-gram via `gensim` Word2Vec) on the GO DAG, producing dense **64-dimensional** vectors stored in `go-terms.emb`. Functionally similar GO terms are placed close together in this space.

**Stage 3 – Protein graph construction.** `training-testing/GNN-Training.ipynb` (cells `DATA PREPROCESS v11.5` and `DATA PREPROCESS v11.0`) maps each protein to its set of GO annotations from the species-specific GAF file. Only **experimentally supported** annotations are kept; entries flagged `IEA` (Inferred from Electronic Annotation) or `ND` (No Data) are discarded, as are GO terms not present in the trained vocabulary. The protein feature vector is the **mean** of its GO term embeddings. The PPI graph keeps only positive interactions with STRING combined score ≥ 700; negatives are used exclusively as supervised link prediction targets and never participate in message passing.

**Stage 4 – GNN training and link prediction.** Three encoders are trained, all sharing an identical 3-layer MLP decoder that takes the concatenation of the two endpoint embeddings and outputs an interaction probability via sigmoid.

| Encoder | Layers | Hidden dim | Notes |
|---|---|---|---|
| GCN | 3 | 128 | Normalized propagation with self-loops, BatchNorm after the first two layers |
| GATv2 | 2 | 256 | First layer uses 8 attention heads (concat), second uses 1 head; BatchNorm after each layer; 2 layers to mitigate over-smoothing |
| GraphSAGE | 3 | 256 | Mean aggregator, BatchNorm after each layer, ReLU + dropout after the first two |

All encoders output 64-dimensional node embeddings. Training uses the Adam optimizer with `ReduceLROnPlateau` (factor=0.5, patience=5), a maximum of 400 epochs, and early stopping when validation AUROC does not improve for 10 consecutive epochs. The validation and test edges are strictly excluded from the message-passing graph to prevent information leakage (transductive link prediction).

### Hyperparameters

The first value applies to the **TransformerGO-derived datasets (v11.5)**, the second (where shown) applies to the **STRING v11.0 benchmark** when dataset-specific tuning was needed for stable convergence.

| Hyperparameter | GCN | GATv2 | GraphSAGE |
|---|---|---|---|
| Learning rate | 1e-3 | 5e-4 | 1e-4 |
| Weight decay | 1e-5 | 1e-6 / 1e-4 | 5e-4 |
| Dropout | 0.1 | 0.15 / 0.20 | 0.3 |
| Attention heads | – | 8 | – |
| Scheduler | ReduceLROnPlateau (patience=5) | | |

## Datasets

Two evaluation regimes are supported.

The **TransformerGO-derived datasets (STRING v11.5)** are constructed by the notebooks under `datasets/v11.5/interaction-datasets/`. Positives are extracted with combined score ≥ 700, and a balanced 1:1 set of negatives is sampled from random protein pairs that do **not** appear in STRING at any confidence level. After filtering for proteins with valid GO annotations the splits are 64% / 16% / 20% (train / validation / test). Final sizes: 243,560 positive and 242,599 negative pairs for human; 115,793 positive and 114,409 negative pairs for yeast.

The **STRING v11.0 benchmark** uses the fixed `no-mirror` splits popularized by Jain & Bader and Kulmanov et al. Negatives here are STRING pairs with confidence < 700 (so they may include low-confidence true interactions). Files are placed under `datasets/v11.0/interaction-datasets/{train,valid,test}/`. After functional filtering: 410,286 positives and 360,795 negatives for human; 113,418 positives and 89,958 negatives for yeast.

## Requirements

The pipeline is written in Python 3 and was tested with the following libraries:

- `torch`, `torch_geometric`
- `networkx`
- `node2vec`, `gensim`
- `numpy`, `pandas`, `scikit-learn`
- `matplotlib`, `seaborn`
- `tqdm`

A typical setup looks like:

```bash
pip install torch torch_geometric networkx node2vec gensim numpy pandas scikit-learn matplotlib seaborn tqdm
```

A CUDA-capable GPU is recommended for training the GNN encoders, especially on the human v11.0 benchmark, which has the largest graph.

## How to Reproduce

The steps below reproduce the v11.0 results for *S. cerevisiae*. Swap `4932` for `9606` to run human, and switch the `v11.0` paths to `v11.5` to run on the updated interactome.

**1. Parse the GO ontology.** Run `term-encoding-module/OBOParsing.ipynb` to convert `datasets/v11.0/go-terms/go-basic.obo` into `go-basic.obo.csv`.

**2. Build the GO graph and helper dictionaries.** Run `term-encoding-module/node2vecTerms.ipynb`. This produces `datasets/v11.0/go-terms/graph/go-terms.edgelist`, `go_id_dict`, and `go_namespace_dict`.

**3. Train GO term embeddings.**

```bash
python term-encoding-module/node2vec-embeddings.py \
    --input  datasets/v11.0/go-terms/graph/go-terms.edgelist \
    --output datasets/v11.0/go-terms/embeddings/go-terms.emb \
    --dimensions 64
```

**4. Build the PPI splits.** Run `datasets/v11.0/interaction-datasets/PPIv11.0.ipynb` (set `org_id` to `4932` or `9606`) to populate the `train/`, `valid/`, and `test/` folders. For the v11.5 protocol, run `datasets/v11.5/interaction-datasets/PPIv11.5.ipynb` instead, which produces the `*.protein.positive.v11.5.txt` and `*.protein.negative.v11.5.txt` files.

**5. Train and evaluate the GNN encoders.** Run `training-testing/GNN-Training.ipynb`. The notebook is organized into self-contained sections: a v11.5 preprocessing block, a v11.0 preprocessing block, and one block per encoder (GCN, GATv2, GraphSAGE). Each training block reports AUROC, AUPRC, accuracy, and F1, and produces precision-recall and ROC plots.

## Results

**STRING v11.5 (TransformerGO-derived datasets).** GATv2 is the top performer on both species, with GCN a close second.

| Organism | Model | AUROC | AUPRC | Accuracy | F1 |
|---|---|---|---|---|---|
| *H. sapiens* | GraphSAGE | 0.9336 | 0.9362 | 0.8639 | 0.8629 |
| *H. sapiens* | GCN | 0.9592 | 0.9648 | 0.9010 | 0.8993 |
| *H. sapiens* | **GATv2** | **0.9617** | **0.9681** | **0.9091** | **0.9082** |
| *S. cerevisiae* | GraphSAGE | 0.9654 | 0.9663 | 0.9094 | 0.9100 |
| *S. cerevisiae* | GCN | 0.9745 | 0.9754 | 0.9261 | 0.9266 |
| *S. cerevisiae* | **GATv2** | **0.9756** | **0.9767** | **0.9308** | **0.9303** |

Both GCN and GATv2 surpass the TransformerGO baseline (0.958 human, 0.973 yeast).

**STRING v11.0 benchmark (state-of-the-art comparison).** Our GATv2 model attains the highest AUROC for both species; GCN is competitive and even slightly better on human AUPRC and accuracy.

| Model | *S. cerevisiae* AUROC | *H. sapiens* AUROC |
|---|---|---|
| Resnik | 0.870 | 0.890 |
| Onto2Vec | 0.800 | 0.770 |
| Opa2Vec | 0.880 | 0.880 |
| EL Embeddings | 0.930 | 0.900 |
| Node2VecCOS | 0.847 | 0.845 |
| Node2VecNN | 0.952 | 0.958 |
| TransformerGO | 0.961 | 0.974 |
| GraphSAGE (ours) | 0.956 | 0.965 |
| GCN (ours) | 0.971 | 0.976 |
| **GATv2 (ours)** | **0.974** | **0.977** |

Detailed metrics on the v11.0 benchmark:

| Organism | Model | AUROC | AUPRC | Accuracy | F1 |
|---|---|---|---|---|---|
| *H. sapiens* | GraphSAGE | 0.9654 | 0.9727 | 0.9125 | 0.9164 |
| *H. sapiens* | GCN | 0.9762 | **0.9824** | **0.9354** | 0.9380 |
| *H. sapiens* | GATv2 | **0.9770** | 0.9820 | 0.9349 | **0.9382** |
| *S. cerevisiae* | GraphSAGE | 0.9563 | 0.9676 | 0.8944 | 0.9026 |
| *S. cerevisiae* | GCN | 0.9714 | 0.9798 | 0.9237 | 0.9298 |
| *S. cerevisiae* | **GATv2** | **0.9735** | **0.9801** | **0.9274** | **0.9337** |

## Methodological Differences from TransformerGO

TransformerGO treats each protein as an unordered set of GO term vectors and uses self- and cross-attention to score pairwise semantic compatibility, without explicitly accessing the broader interaction network during inference. Our framework instead aggregates a protein's GO term embeddings upstream into a single fixed-length feature vector (d=64), inserts that vector as a node attribute in the PPI graph, and lets the GNN encoder propagate it across high-confidence training edges. This trades some term-level interpretability for explicit "guilt-by-association" topological reasoning, which the empirical results show to be the dominant signal on dense, modern interactomes.

## Citation

If you use this framework, please cite:

```
Pantelis Makrygiannis, Nikitas-Rigas Kalogeropoulos, Agorakis Bompotas, Christos Makris.
"A Graph-based Approach to Efficiently Predicting Protein-Protein Interactions."
International Journal on Artificial Intelligence Tools (IJAIT).
```

## Contact

- Pantelis Makrygiannis – st1067526@ceid.upatras.gr
- Nikitas-Rigas Kalogeropoulos – kalogeropo@ceid.upatras.gr
- Agorakis Bompotas – mpompotas@ceid.upatras.gr
- Christos Makris – makri@ceid.upatras.gr

Department of Computer Engineering and Informatics, University of Patras, University Campus, Rio, Greece.
