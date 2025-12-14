<div align="center">
  <img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=1a1a2e&height=120&section=header"/>
  
  <h1>
    <img src="https://readme-typing-svg.herokuapp.com/?lines=Benchmark+de+Clustering+NLP;Embeddings+Transformer;PCA+%2B+Spectral+Clustering;Aprendizado+N%C3%A3o+Supervisionado&font=Fira+Code&center=true&width=600&height=50&color=4A90E2&vCenter=true&pause=1000&size=24" />
  </h1>
  
  <samp>UFRN · Engenharia Elétrica · Projeto de Clustering NLP</samp>
  <br/><br/>
  
  <a href="README.md">
    <img src="https://img.shields.io/badge/Lang-English-blue?style=for-the-badge&logo=google-translate&logoColor=white" alt="Read in English"/>
  </a>
  <br/><br/>
  
  <img src="https://img.shields.io/badge/Python-3.12-4A90E2?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/BioBERT-SBERT-EE4C2C?style=for-the-badge&logo=huggingface&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3.0-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Concluído-00C853?style=for-the-badge"/>
</div>

<br/>

## `> visao_geral_projeto()`

```python
class NLPClusteringProject:
    def __init__(self):
        self.titulo = "Benchmark de Modelos de Embedding & Algoritmos de Clustering"
        self.subtitulo = "Aprendizado Não Supervisionado em Notícias e Textos Curtos"
        self.datasets = ["20Newsgroups (20NG-6)", "Textos Curtos em Português (PT-6)"]
        self.instituicao = "UFRN - Universidade Federal do Rio Grande do Norte"
        self.departamento = "Centro de Tecnologia - Eng. Elétrica"
        self.autor = "Cauã Vitor Figueredo Silva"
        self.data = "Dezembro 2025"
        self.python_version = "3.12"
    
    def pipeline(self):
        return {
            "embeddings": ["TF-IDF+SVD", "SBERT", "GTE-Base", "BGE-M3"],
            "reducao": "PCA (768d -> 100d) para embeddings densos",
            "algoritmos": ["KMeans", "GMM", "Agglomerative", "DBSCAN", "Spectral", "HDBSCAN"],
            "metricas": ["ARI", "NMI", "Purity", "Silhouette"]
        }
    
    def otimizacao_performance(self):
        return [
            "Redução de Dimensionalidade (PCA)",
            "Spectral Clustering com Vizinhos Mais Próximos (Grafo Esparso)",
            "Processamento Paralelo (n_jobs=-1)",
            "Sistema de Checkpoint (Tolerância a Falhas)"
        ]
    
    def resultados_finais(self):
        return {
            "melhor_embedding": "BGE-M3 (BAAI)",
            "melhor_algoritmo": "K-Means / GMM",
            "melhor_score_pt6": {"ARI": 0.94, "NMI": 0.93},
            "conclusao": "Embeddings densos com PCA superam features brutas significativamente."
        }
```

<br/>

## `> tecnologias`

<div align="center">
  <img src="https://skillicons.dev/icons?i=python,sklearn,pytorch,git,github,vscode&theme=dark&perline=6" />
</div>

<table align="center">
<tr>
<td align="center" width="33%">
<strong>🧠 Embeddings & Transformers</strong><br/><br/>
<img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black"/>
<img src="https://img.shields.io/badge/Sentence--BERT-v2-00C853?style=flat-square"/>
<img src="https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white"/>
</td>
<td align="center" width="33%">
<strong>📊 Pipeline de Clustering</strong><br/><br/>
<img src="https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/UMAP-0.5-4A90E2?style=flat-square"/>
<img src="https://img.shields.io/badge/HDBSCAN-0.8-EE4C2C?style=flat-square"/>
<img src="https://img.shields.io/badge/Matplotlib-3.8-11557c?style=flat-square"/>
</td>
<td align="center" width="33%">
<strong>🔧 Desenvolvimento</strong><br/><br/>
<img src="https://img.shields.io/badge/Jupyter-Lab-F37626?style=flat-square&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-2.1-150458?style=flat-square&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white"/>
</td>
</tr>
</table>

<br/>

## `> estrutura_tprojeto`

```
nlp-clustering-benchmark/
│
├── 📊 data/
│   ├── raw/                    # CSVs brutos e Datasets
│   ├── embeddings/             # Embeddings em cache (.npy)
│   └── processed/              # Dados pré-processados
│
├── 🧠 src/
│   ├── config.py               # Configuração global e Hiperparâmetros
│   ├── utils.py                # Funções auxiliares (PCA, Métricas, Plotagem)
│   └── __init__.py
│
├── 📓 notebooks/
│   ├── 01_data_prep.ipynb      # Limpeza e pré-processamento de dados
│   ├── 02_embeddings.ipynb     # Geração de Embeddings (SBERT, GTE, BGE)
│   ├── 03_clustering.ipynb     # Loop Principal: Clustering e Avaliação (com Checkpoint)
│   └── 04_visualization.ipynb  # Projeções 2D (PCA, t-SNE, UMAP)
│
├── 💾 results/
│   ├── tables/                 # Resultados CSV (clustering_results.csv)
│   └── figures/                # Gráficos gerados (Heatmaps, Scatter plots)
│       ├── clustering_metrics_heatmap.png
│       ├── compare_pt6_bge_umap.png
│       └── proj_20ng6_bge_umap.png
│
├── .gitignore
└── README.md
```

<br/>

## `> metodologia`

<table align="center">
<tr>
<td width="50%">
<h3 align="center">📉 Redução de Dimensionalidade (PCA)</h3>
<p align="center">
<img src="https://img.shields.io/badge/PCA-100_Componentes-4A90E2?style=for-the-badge"/>
</p>
<p><samp>Embeddings de alta dimensão (768d/1024d) sofrem da "Maldição da Dimensionalidade", tornando métricas de distância menos eficazes e lentificando algoritmos. Reduzimos para <strong>100 dimensões</strong> preservando via PCA 95%+ da variância.</samp></p>

**Benefícios Chave:**
- **Velocidade**: Execução 10x mais rápida para Spectral/DBSCAN.
- **Qualidade**: Remove ruído, melhorando a coesão dos clusters.
</td>
<td width="50%">
<h3 align="center">⚡ Otimização do Spectral Clustering</h3>
<p align="center">
<img src="https://img.shields.io/badge/Affinity-Vizinhos_Mais_Proximos-00C853?style=for-the-badge"/>
</p>
<p><samp>O kernel RBF padrão constrói uma matriz densa $N \times N$ ($O(N^2)$ memória). Mudamos para <strong>Vizinhos Mais Próximos</strong> para construir um grafo esparso.</samp></p>

**Otimização Crítica:**
```python
# Antes (Crash/Travamento)
affinity="rbf"

# Depois (Segundos)
affinity="nearest_neighbors", n_neighbors=10
```
</td>
</tr>
</table>

<br/>

## `> visualizacao_resultados`

<div align="center">

### 🏆 Melhor Modelo: Embedding BGE + K-Means/GMM

Alcançamos resultados de estado da arte no dataset PT-6 usando o embedding **BGE-M3**.

<table>
<tr>
<td align="center">
<strong>Comparação: Ground Truth vs Predito</strong><br/>
<img src="results/figures/compare_pt6_bge_umap.png" width="100%"/><br/>
<em>Esquerda: Classes Reais | Direita: Clusters K-Means (Correspondência Perfeita!)</em>
</td>
</tr>
</table>

### 📊 Heatmaps de Métricas (Visão Geral de Performance)

<table>
<tr>
<td align="center">
<strong>Scores ARI & NMI entre Embeddings</strong><br/>
<img src="results/figures/clustering_metrics_heatmap.png" width="100%"/><br/>
<em>Vermelho Mais Escuro = Melhor Performance. Note como o BGE domina.</em>
</td>
</tr>
</table>

</div>

<br/>

## `> resumo_resultados`

<table align="center">
<tr>
<td width="50%" align="center">
<h3>📊 Dataset PT-6 (Textos Curtos)</h3>

| Métrica | Melhor Valor | Modelo |
|--------|-------|-------|
| **ARI** | **0.941** | BGE + K-Means |
| **NMI** | **0.935** | BGE + K-Means |
| **Purity** | **0.974** | BGE + GMM |
| **Silhouette** | **0.224** | SBERT + DBSCAN |

<img src="https://img.shields.io/badge/Resultado-Quase_Perfeito-00C853?style=flat-square"/>
</td>
<td width="50%" align="center">
<h3>📊 Dataset 20NG-6 (Notícias)</h3>

| Métrica | Melhor Valor | Modelo |
|--------|-------|-------|
| **ARI** | **0.60** | BGE + GMM |
| **NMI** | **0.66** | BGE + GMM |
| **Purity** | **0.78** | BGE + K-Means |
| **Silhouette** | **0.15** | TFIDF + DBSCAN |

<img src="https://img.shields.io/badge/Resultado-Baseline_Solido-4A90E2?style=flat-square"/>
</td>
</tr>
</table>

<br/>

## `> execucao`

```bash
# Clone o repositório
git clone https://github.com/takaokensei/nlp-clustering-benchmark.git
cd nlp-clustering-benchmark

# Crie o ambiente virtual (uv ou venv)
uv venv .venv
.venv\Scripts\activate

# Instale dependências
pip install -r requirements.txt

# Execute o pipeline (na ordem)
# 1. Preparação de Dados
jupyter notebook notebooks/01_data_prep.ipynb

# 2. Geração de Embeddings (Em Cache)
jupyter notebook notebooks/02_embeddings.ipynb

# 3. Clustering (O Benchmark Principal)
jupyter notebook notebooks/03_clustering.ipynb

# 4. Visualização
jupyter notebook notebooks/04_visualization.ipynb
```

<br/>

## `> contato`

<div align="center">
  
  <strong>Cauã Vitor Figueredo Silva</strong>
  <br/>
  <samp>Estudante de Engenharia Elétrica</samp>
  <br/>
  <samp>UFRN - Universidade Federal do Rio Grande do Norte</samp>
  
  <br/><br/>
  
  <a href="https://github.com/takaokensei">
    <img src="https://img.shields.io/badge/-GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/>
  </a>
</div>

<br/>

<div align="center">
  <img src="https://img.shields.io/badge/Feito_com-Python_3.12-EE4C2C?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Licenca-MIT-1a1a2e?style=for-the-badge"/>
</div>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=1a1a2e&height=100&section=footer"/>
