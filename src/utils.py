"""
Funções auxiliares para o projeto NLP Clustering Benchmark.
Inclui cálculo de métricas, carregamento de dados e visualizações.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    confusion_matrix,
)
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.config import (
    RANDOM_STATE,
    FIGURE_CONFIG,
    PLOT_STYLE,
    COLOR_PALETTE,
    TABLES_DIR,
)


# ============================================================================
# MÉTRICAS DE AVALIAÇÃO
# ============================================================================


def compute_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula a Pureza (Purity) dos clusters.
    
    Pureza mede a proporção de elementos que estão no cluster majoritário
    de cada cluster em relação ao total de elementos.
    
    Args:
        y_true: Rótulos verdadeiros (ground truth)
        y_pred: Rótulos de cluster atribuídos
        
    Returns:
        Valor de pureza entre 0 e 1 (quanto maior, melhor)
    """
    confusion_mat = confusion_matrix(y_true, y_pred)
    # Para cada cluster, pegar o número de elementos da classe majoritária
    cluster_purities = np.max(confusion_mat, axis=0)
    # Pureza total = soma das purezas de cada cluster / total de elementos
    purity = np.sum(cluster_purities) / np.sum(confusion_mat)
    return float(purity)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calcula todas as métricas de avaliação (externas e internas).
    
    Args:
        y_true: Rótulos verdadeiros
        y_pred: Rótulos de cluster atribuídos
        X: Matriz de features (opcional, necessário para Silhouette)
        
    Returns:
        Dicionário com todas as métricas calculadas
    """
    metrics = {}
    
    # Métricas externas (requerem y_true)
    metrics["ari"] = adjusted_rand_score(y_true, y_pred)
    metrics["nmi"] = normalized_mutual_info_score(y_true, y_pred)
    metrics["purity"] = compute_purity(y_true, y_pred)
    
    # Métrica interna (requer X)
    if X is not None:
        try:
            # Silhouette pode falhar se houver apenas 1 cluster ou muitos clusters vazios
            n_clusters = len(np.unique(y_pred))
            n_samples = len(y_pred)
            
            if n_clusters > 1 and n_clusters < n_samples:
                # Para datasets grandes (>5000), usar amostra para acelerar
                if n_samples > 5000:
                    # Amostrar 5000 pontos aleatórios
                    sample_indices = np.random.RandomState(RANDOM_STATE).choice(
                        n_samples, size=5000, replace=False
                    )
                    metrics["silhouette"] = silhouette_score(
                        X[sample_indices], 
                        y_pred[sample_indices], 
                        metric="euclidean", 
                        random_state=RANDOM_STATE
                    )
                else:
                    metrics["silhouette"] = silhouette_score(
                        X, y_pred, metric="euclidean", random_state=RANDOM_STATE
                    )
            else:
                metrics["silhouette"] = np.nan
        except Exception as e:
            print(f"Erro ao calcular Silhouette: {e}")
            metrics["silhouette"] = np.nan
    else:
        metrics["silhouette"] = np.nan
    
    return metrics


# ============================================================================
# CARREGAMENTO E PERSISTÊNCIA
# ============================================================================


def save_embedding(
    embedding: np.ndarray,
    dataset_name: str,
    embedding_type: str,
    embeddings_dir: Path,
) -> Path:
    """
    Salva embedding em arquivo .npy.
    
    Args:
        embedding: Matriz de embeddings (n_samples, n_features)
        dataset_name: Nome do dataset ('pt6' ou '20ng6')
        embedding_type: Tipo de embedding ('tfidf_svd', 'sbert', 'gte', 'bge')
        embeddings_dir: Diretório onde salvar
        
    Returns:
        Caminho do arquivo salvo
    """
    filename = f"{dataset_name}_{embedding_type}.npy"
    filepath = embeddings_dir / filename
    np.save(filepath, embedding)
    print(f"Embedding salvo em: {filepath}")
    return filepath


def load_embedding(
    dataset_name: str,
    embedding_type: str,
    embeddings_dir: Path,
) -> Optional[np.ndarray]:
    """
    Carrega embedding de arquivo .npy se existir.
    
    Args:
        dataset_name: Nome do dataset ('pt6' ou '20ng6')
        embedding_type: Tipo de embedding ('tfidf_svd', 'sbert', 'gte', 'bge')
        embeddings_dir: Diretório onde buscar
        
    Returns:
        Matriz de embeddings ou None se não existir
    """
    filename = f"{dataset_name}_{embedding_type}.npy"
    filepath = embeddings_dir / filename
    
    if filepath.exists():
        embedding = np.load(filepath)
        print(f"Embedding carregado de: {filepath}")
        return embedding
    else:
        print(f"Embedding não encontrado: {filepath}")
        return None


def detect_class_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Detecta dinamicamente o nome da coluna de classe no DataFrame.
    Prioriza colunas com nomes textuais (object) sobre numéricas (int).
    A comparação é case-insensitive.
    
    Args:
        df: DataFrame com os dados
        candidates: Lista de nomes candidatos para a coluna de classe
        
    Returns:
        Nome da coluna encontrada ou None
    """
    # Criar dicionário mapeando nomes em minúsculas para nomes originais
    column_lower_map = {col.lower(): col for col in df.columns}
    
    # Primeiro, verificar candidatos explícitos (case-insensitive)
    for candidate in candidates:
        candidate_lower = candidate.lower()
        if candidate_lower in column_lower_map:
            return column_lower_map[candidate_lower]
    
    # Se não encontrar, procurar por colunas que parecem ser de classe
    # Priorizar colunas com valores textuais (object) sobre numéricas
    text_columns = []
    numeric_columns = []
    
    for col in df.columns:
        if df[col].dtype in ["object", "int64", "int32"]:
            n_unique = df[col].nunique()
            if 2 <= n_unique <= 20:  # Número razoável de classes
                if df[col].dtype == "object":
                    text_columns.append(col)
                else:
                    numeric_columns.append(col)
    
    # Preferir colunas textuais (com nomes) sobre numéricas
    if text_columns:
        print(f"Coluna de classe detectada automaticamente: {text_columns[0]}")
        return text_columns[0]
    elif numeric_columns:
        print(f"Coluna de classe detectada automaticamente: {numeric_columns[0]}")
        return numeric_columns[0]
    
    return None


# ============================================================================
# VISUALIZAÇÃO
# ============================================================================


def plot_2d_comparison(
    X_2d: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method_name: str,
    embedding_name: str,
    dataset_name: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    Gera visualização 2D lado a lado: Classe Real vs. Cluster Atribuído.
    
    Args:
        X_2d: Matriz 2D reduzida (n_samples, 2)
        y_true: Rótulos verdadeiros
        y_pred: Rótulos de cluster
        method_name: Nome do método de redução ('PCA', 't-SNE', 'UMAP')
        embedding_name: Nome do embedding
        dataset_name: Nome do dataset
        save_path: Caminho para salvar a figura (opcional)
    """
    plt.style.use(PLOT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_CONFIG["figsize"])
    
    # Plot 1: Classe Real
    scatter1 = axes[0].scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=y_true,
        cmap=COLOR_PALETTE,
        alpha=0.6,
        s=20,
    )
    axes[0].set_title(f"{method_name} - Classe Real\n{embedding_name} ({dataset_name})")
    axes[0].set_xlabel(f"{method_name} Component 1")
    axes[0].set_ylabel(f"{method_name} Component 2")
    plt.colorbar(scatter1, ax=axes[0])
    
    # Plot 2: Cluster Atribuído
    scatter2 = axes[1].scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=y_pred,
        cmap=COLOR_PALETTE,
        alpha=0.6,
        s=20,
    )
    axes[1].set_title(f"{method_name} - Cluster Atribuído\n{embedding_name} ({dataset_name})")
    axes[1].set_xlabel(f"{method_name} Component 1")
    axes[1].set_ylabel(f"{method_name} Component 2")
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=FIGURE_CONFIG["dpi"], format=FIGURE_CONFIG["format"])
        print(f"Figura salva em: {save_path}")
    
    plt.close()


def create_results_dataframe(
    results: List[Dict],
) -> pd.DataFrame:
    """
    Cria DataFrame consolidado com todos os resultados de métricas.
    
    Args:
        results: Lista de dicionários com resultados de cada combinação
        
    Returns:
        DataFrame organizado com métricas
    """
    df = pd.DataFrame(results)
    
    # Reorganizar colunas para melhor visualização
    column_order = ["dataset", "embedding", "algorithm"] + [
        col for col in df.columns if col not in ["dataset", "embedding", "algorithm"]
    ]
    
    # Garantir que todas as colunas existam
    available_cols = [col for col in column_order if col in df.columns]
    df = df[available_cols]
    
    return df


def save_results_table(
    df: pd.DataFrame,
    filename: str,
    tables_dir: Path,
) -> Path:
    """
    Salva tabela de resultados em CSV.
    
    Args:
        df: DataFrame com resultados
        filename: Nome do arquivo (sem extensão)
        tables_dir: Diretório onde salvar
        
    Returns:
        Caminho do arquivo salvo
    """
    filepath = tables_dir / f"{filename}.csv"
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"Tabela salva em: {filepath}")
    return filepath

