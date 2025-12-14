"""
Configurações centralizadas do projeto NLP Clustering Benchmark.
Define seeds, modelos, parâmetros e caminhos de arquivos.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Criar diretórios se não existirem
for dir_path in [RAW_DATA_DIR, EMBEDDINGS_DIR, FIGURES_DIR, TABLES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# REPRODUTIBILIDADE
# ============================================================================
RANDOM_STATE = 42
N_INIT_KMEANS = 10  # Número de inicializações para K-Means
PCA_N_COMPONENTS = 100  # Número de componentes para redução de dimensionalidade
TIMEOUT_SECONDS = 300  # Tempo máximo por algoritmo (5 minutos)

# ============================================================================
# EMBEDDINGS - Modelos e Parâmetros
# ============================================================================

# TF-IDF + SVD
TFIDF_CONFIG = {
    "ngram_range": (1, 2),  # Pode ser (1, 3) também
    "max_features": 50000,
    "min_df": 2,
    "max_df": 0.95,
}

SVD_CONFIG = {
    "n_components": 300,
    "random_state": RANDOM_STATE,
}

# Modelos Sentence Transformers
EMBEDDING_MODELS = {
    "sbert": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "gte": "thenlper/gte-base",  # Modelo GTE base (requer autenticação no Hugging Face)
    "bge": "BAAI/bge-m3",
}

# Configurações de batch para embeddings
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DEVICE = "cpu"  # Mude para "cuda" se tiver GPU

# ============================================================================
# CLUSTERING - Algoritmos e Parâmetros
# ============================================================================

N_CLUSTERS = 6  # Número de classes nas bases PT-6 e 20NG-6

CLUSTERING_CONFIGS = {
    "kmeans": {
        "n_clusters": N_CLUSTERS,
        "init": "k-means++",
        "n_init": N_INIT_KMEANS,
        "random_state": RANDOM_STATE,
        "max_iter": 300,
    },
    "gmm": {
        "n_components": N_CLUSTERS,
        "random_state": RANDOM_STATE,
        "n_init": N_INIT_KMEANS,
        "max_iter": 100,
    },
    "agglomerative": {
        "n_clusters": N_CLUSTERS,
        "linkage": "ward",  # Pode ser "complete" também
    },
    "dbscan": {
        # Parâmetros serão determinados via busca (k-distance graph ou grid search)
        # Valores padrão - devem ser ajustados por dataset/embedding
        "eps": 0.5,
        "min_samples": 5,
        "metric": "euclidean",
        "n_jobs": -1,
    },
    "spectral": {
        "n_clusters": N_CLUSTERS,
        "random_state": RANDOM_STATE,
        "affinity": "nearest_neighbors",
        "n_neighbors": 10,
        "n_jobs": -1,
    },
    "hdbscan": {
        "min_cluster_size": 10,
        "min_samples": 5,
        "metric": "euclidean",
        "core_dist_n_jobs": -1,
    },
}

# ============================================================================
# REDUÇÃO DIMENSIONAL - Visualização
# ============================================================================

REDUCTION_CONFIGS = {
    "pca": {
        "n_components": 2,
        "random_state": RANDOM_STATE,
    },
    "tsne": {
        "n_components": 2,
        "random_state": RANDOM_STATE,
        "perplexity": 30,
        "n_iter": 1000,
        "learning_rate": "auto",
    },
    "umap": {
        "n_components": 2,
        "random_state": RANDOM_STATE,
        "n_neighbors": 15,
        "min_dist": 0.1,
        "metric": "euclidean",
    },
}

# ============================================================================
# DATASETS
# ============================================================================

# 20 Newsgroups - 6 categorias selecionadas
TWENTY_NG_CATEGORIES = [
    "comp.graphics",
    "comp.sys.mac.hardware",
    "rec.autos",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.med",
]

# PT-6 - Nome da coluna de classe (será detectado dinamicamente)
# Priorizar "Categoria" pois contém nomes das classes (mais intuitivo)
PT6_CLASS_COLUMN_CANDIDATES = ["categoria", "category", "classe", "class", "label"]

# ============================================================================
# MÉTRICAS
# ============================================================================

METRICS_TO_COMPUTE = ["ari", "nmi", "purity", "silhouette"]

# ============================================================================
# VISUALIZAÇÃO
# ============================================================================

FIGURE_CONFIG = {
    "dpi": 300,  # Alta resolução para relatório
    "figsize": (12, 5),  # Lado a lado: (largura, altura)
    "format": "png",
}

PLOT_STYLE = "seaborn-v0_8"
COLOR_PALETTE = "Set2"  # Paleta de cores para clusters

