
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

# Cell 1: Title and imports
text_cell_1 = """# 04 - Visualizações 2D

Este notebook gera visualizações 2D (PCA, t-SNE, UMAP) para análise qualitativa dos clusters.
Ele compara:
1. **Rótulos Verdadeiros** (Ground Truth)
2. **Clusters Preditos** (pelo melhor algoritmo identificado)
"""

code_cell_1 = """import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Adicionar src ao path
project_root = Path().resolve().parent
sys.path.append(str(project_root))

from src.config import (
    EMBEDDINGS_DIR, REDUCTION_CONFIGS, FIGURES_DIR, RANDOM_STATE
)
from src.utils import load_embedding, plot_2d_comparison, load_checkpoint_results, TABLES_DIR

# Configurar estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")
"""

# Cell 2: Load Data (True Labels)
text_cell_2 = """## 1. Carregar Dados e Labels Reais"""

code_cell_2 = """# Carregar labels verdadeiros (mesma lógica do nb 03)
from sklearn.datasets import fetch_20newsgroups
from src.config import TWENTY_NG_CATEGORIES, RAW_DATA_DIR, PT6_CLASS_COLUMN_CANDIDATES
from src.utils import detect_class_column

# 20NG
newsgroups = fetch_20newsgroups(subset='all', categories=TWENTY_NG_CATEGORIES, remove=('headers', 'footers', 'quotes'))
y_true_20ng = newsgroups.target
target_names_20ng = newsgroups.target_names

# PT6
df_pt6 = pd.read_csv(RAW_DATA_DIR / "pt6_preprocessed.csv", encoding='utf-8-sig')
class_col = detect_class_column(df_pt6, PT6_CLASS_COLUMN_CANDIDATES)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_true_pt6 = le.fit_transform(df_pt6[class_col])
target_names_pt6 = le.classes_

ground_truth = {
    '20ng6': {'y': y_true_20ng, 'names': target_names_20ng},
    'pt6': {'y': y_true_pt6, 'names': target_names_pt6}
}
print("✅ Labels verdadeiros carregados.")
"""

# Cell 3: Load Embeddings and Compute Projections
text_cell_3 = """## 2. Carregar Embeddings e Calcular Projeções 2D

Vamos calcular PCA, t-SNE e UMAP para todos os embeddings. Isso pode demorar um pouco.
Recalculamos aqui para garantir que temos as projeções exatas para plotagem.
"""

code_cell_3 = """import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

embeddings_map = {
    '20ng6': {},
    'pt6': {}
}

projections = {
    '20ng6': {},
    'pt6': {}
}

embedding_types = ['tfidf_svd', 'sbert', 'gte', 'bge']

for dataset in ['20ng6', 'pt6']:
    print(f"\\n📊 Processando {dataset}...")
    for emb_type in embedding_types:
        # Carregar embedding
        X = load_embedding(dataset, emb_type, EMBEDDINGS_DIR)
        if X is None:
            continue
            
        embeddings_map[dataset][emb_type] = X
        projections[dataset][emb_type] = {}
        
        # 1. PCA
        print(f"   📉 PCA -> {emb_type}")
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        projections[dataset][emb_type]['pca'] = pca.fit_transform(X)
        
        # 2. t-SNE (usar init='pca' para estabilidade)
        print(f"   📉 t-SNE -> {emb_type}")
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE, init='pca', learning_rate='auto')
        projections[dataset][emb_type]['tsne'] = tsne.fit_transform(X)
        
        # 3. UMAP
        print(f"   📉 UMAP -> {emb_type}")
        reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
        projections[dataset][emb_type]['umap'] = reducer.fit_transform(X)

print("\\n✅ Todas as projeções calculadas!")
"""

# Cell 4: Visualization Function
text_cell_4 = """## 3. Visualizar Comparação: Ground Truth vs Clusters

Vamos focar no melhor algoritmo (identificado no passo anterior, geralmente KMeans ou GMM) e no melhor embedding (BGE).
"""

code_cell_4 = """def plot_projections(dataset_name, emb_type, method='umap'):
    X_proj = projections[dataset_name][emb_type][method]
    y_true = ground_truth[dataset_name]['y']
    target_names = ground_truth[dataset_name]['names']
    
    # Plot apenas do Ground Truth por enquanto (para validar separação)
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y_true, cmap='tab10', alpha=0.6, s=10)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(target_names), title="Classes")
    plt.title(f'Projeção {method.upper()} - {dataset_name.upper()} - {emb_type} (Ground Truth)')
    plt.xlabel(f'{method.upper()} 1')
    plt.ylabel(f'{method.upper()} 2')
    
    filename = FIGURES_DIR / f"proj_{dataset_name}_{emb_type}_{method}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Salvou em: {filename}")

# Exemplo: Plotar UMAP do BGE para ambos datasets
plot_projections('pt6', 'bge', 'umap')
plot_projections('20ng6', 'bge', 'umap')
"""

# Cell 5: Advanced Comparison (Side-by-Side with Clusters)
text_cell_5 = """## 4. Comparação Lado a Lado: Real vs Predito (KMeans)

Vamos carregar os resultados do passo 3 (ou rodar KMeans aqui rapidamente) para comparar visualmente.
"""

code_cell_5 = """from sklearn.cluster import KMeans

def plot_side_by_side(dataset_name, emb_type, method='umap'):
    # Dados
    X = embeddings_map[dataset_name][emb_type]
    X_proj = projections[dataset_name][emb_type][method]
    y_true = ground_truth[dataset_name]['y']
    
    # Rodar KMeans rápido para pegar labels preditos
    kmeans = KMeans(n_clusters=6, n_init=10, random_state=RANDOM_STATE)
    y_pred = kmeans.fit_predict(X)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Real
    scatter1 = axes[0].scatter(X_proj[:, 0], X_proj[:, 1], c=y_true, cmap='tab10', alpha=0.5, s=15)
    axes[0].set_title(f'Ground Truth ({dataset_name}/{emb_type})')
    axes[0].legend(*scatter1.legend_elements(), title="Classes")
    
    # Plot 2: Predito
    scatter2 = axes[1].scatter(X_proj[:, 0], X_proj[:, 1], c=y_pred, cmap='tab10', alpha=0.5, s=15)
    axes[1].set_title(f'K-Means Clustering ({dataset_name}/{emb_type})')
    
    plt.suptitle(f'Comparação {method.upper()} - {dataset_name.upper()}', fontsize=16)
    plt.tight_layout()
    
    filename = FIGURES_DIR / f"compare_{dataset_name}_{emb_type}_{method}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# Gerar para as melhores combinações
plot_side_by_side('pt6', 'bge', 'umap')
plot_side_by_side('20ng6', 'bge', 'umap')
plot_side_by_side('pt6', 'sbert', 'tsne')
"""

cells = [
    nbf.v4.new_markdown_cell(text_cell_1),
    nbf.v4.new_code_cell(code_cell_1),
    nbf.v4.new_markdown_cell(text_cell_2),
    nbf.v4.new_code_cell(code_cell_2),
    nbf.v4.new_markdown_cell(text_cell_3),
    nbf.v4.new_code_cell(code_cell_3),
    nbf.v4.new_markdown_cell(text_cell_4),
    nbf.v4.new_code_cell(code_cell_4),
    nbf.v4.new_markdown_cell(text_cell_5),
    nbf.v4.new_code_cell(code_cell_5),
]

nb['cells'] = cells

with open('c:/nlp-clustering-benchmark/notebooks/04_visualization.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook 04_visualization.ipynb criado com sucesso!")
