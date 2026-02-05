<div align="center">
  <img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=1a1a2e&height=120&section=header"/>
  
  <h1>
    <img src="https://readme-typing-svg.herokuapp.com/?lines=NLP+Clustering+Benchmark;Transformer+Embeddings;PCA+%2B+Spectral+Clustering;Unsupervised+Learning&font=Fira+Code&center=true&width=600&height=50&color=4A90E2&vCenter=true&pause=1000&size=24" />
  </h1>
  
  <samp>UFRN · Electrical Engineering · NLP Clustering Project</samp>
  <br/><br/>
  
  <img src="https://img.shields.io/badge/Python-3.12-4A90E2?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/BioBERT-SBERT-EE4C2C?style=for-the-badge&logo=huggingface&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3.0-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Complete-00C853?style=for-the-badge"/>
  <br/><br/>
  <a href="README.pt-br.md">
    <img src="https://img.shields.io/badge/Lang-Português-green?style=for-the-badge&logo=google-translate&logoColor=white" alt="Ler em Português"/>
  </a>
</div>

<br/>

## `> project.overview()`

```python
class NLPClusteringProject:
    def __init__(self):
        self.title = "Embedding Models & Clustering Algorithms Benchmark"
        self.subtitle = "Unsupervised Learning on News & Short Texts"
        self.datasets = ["20Newsgroups (20NG-6)", "Portuguese Short Texts (PT-6)"]
        self.institution = "UFRN - Federal University of Rio Grande do Norte"
        self.department = "Center of Technology - Electrical Engineering Dept."
        self.author = "Cauã Vitor Figueredo Silva"
        self.date = "December 2025"
        self.python_version = "3.12"
    
    def pipeline(self):
        return {
            "embeddings": ["TF-IDF+SVD", "SBERT", "GTE-Base", "BGE-M3"],
            "reduction": "PCA (768d -> 100d) for dense embeddings",
            "algorithms": ["KMeans", "GMM", "Agglomerative", "DBSCAN", "Spectral", "HDBSCAN"],
            "metrics": ["ARI", "NMI", "Purity", "Silhouette"]
        }
    
    def performance_optimization(self):
        return [
            "Dimensionality Reduction (PCA)",
            "Spectral Clustering with Nearest Neighbors (Sparse Graph)",
            "Parallel Processing (n_jobs=-1)",
            "Checkpointing System (Fault Tolerance)"
        ]
    
    def final_results(self):
        return {
            "best_embedding": "BGE-M3 (BAAI)",
            "best_model": "K-Means / GMM",
            "best_score_pt6": {"ARI": 0.94, "NMI": 0.93},
            "conclusion": "Dense embeddings with PCA outperform raw features significantly."
        }
```

<br/>

## `> tech_stack`

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
<strong>📊 Clustering Pipeline</strong><br/><br/>
<img src="https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/UMAP-0.5-4A90E2?style=flat-square"/>
<img src="https://img.shields.io/badge/HDBSCAN-0.8-EE4C2C?style=flat-square"/>
<img src="https://img.shields.io/badge/Matplotlib-3.8-11557c?style=flat-square"/>
</td>
<td align="center" width="33%">
<strong>🔧 Development</strong><br/><br/>
<img src="https://img.shields.io/badge/Jupyter-Lab-F37626?style=flat-square&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-2.1-150458?style=flat-square&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white"/>
</td>
</tr>
</table>

<br/>

## `> project_structure`

```
nlp-clustering-benchmark/
│
├── 📊 data/
│   ├── raw/                    # Raw CSVs and Datasets
│   ├── embeddings/             # Cached embeddings (.npy)
│   └── processed/              # Preprocessed data
│
├── 🧠 src/
│   ├── config.py               # Global configuration & Hyperparameters
│   ├── utils.py                # Helper functions (PCA, Metrics, Plotting)
│   └── __init__.py
│
├── 📓 notebooks/
│   ├── 01_data_prep.ipynb      # Data cleaning & preprocessing
│   ├── 02_embeddings.ipynb     # Embedding generation (SBERT, GTE, BGE)
│   ├── 03_clustering.ipynb     # Main Loop: Clustering & Evaluation (with Checkpoint)
│   └── 04_visualization.ipynb  # 2D Projections (PCA, t-SNE, UMAP)
│
├── 💾 results/
│   ├── tables/                 # CSV results (clustering_results.csv)
│   └── figures/                # Generated plots (Heatmaps, Scatter plots)
│       ├── clustering_metrics_heatmap.png
│       ├── compare_pt6_bge_umap.png
│       └── proj_20ng6_bge_umap.png
│
├── .gitignore
└── README.md
```

<br/>

## `> methodology`

<table align="center">
<tr>
<td width="50%">
<h3 align="center">📉 Dimensionality Reduction (PCA)</h3>
<p align="center">
<img src="https://img.shields.io/badge/PCA-100_Components-4A90E2?style=for-the-badge"/>
</p>
<p><samp>High-dimensional embeddings (768d/1024d) cause the "Curse of Dimensionality", making distance metrics less effective and slowing down algorithms. We reduce them to <strong>100 dimensions</strong> preserving 95%+ variance.</samp></p>

**Key Benefits:**
- **Speed**: 10x faster execution for Spectral/DBSCAN.
- **Quality**: Removes noise, improving clustering cohesion.
</td>
<td width="50%">
<h3 align="center">⚡ Spectral Clustering Optimization</h3>
<p align="center">
<img src="https://img.shields.io/badge/Affinity-Nearest_Neighbors-00C853?style=for-the-badge"/>
</p>
<p><samp>Standard RBF kernel constructs a dense $N \times N$ matrix ($O(N^2)$ memory). We switched to <strong>Nearest Neighbors</strong> to build a sparse graph.</samp></p>

**Critical Optimization:**
```python
# Before (Crash/Hang)
affinity="rbf"

# After (Seconds)
affinity="nearest_neighbors", n_neighbors=10
```
</td>
</tr>
</table>

<br/>

## `> results_visualization`

<div align="center">

### 🏆 Best Model: BGE Embedding + K-Means/GMM

We achieved state-of-the-art results on the PT-6 dataset using the **BGE-M3** embedding.

<table>
<tr>
<td align="center">
<strong>Comparison: Ground Truth vs Predicted</strong><br/>
<img src="results/figures/compare_pt6_bge_umap.png" width="100%"/><br/>
<em>Left: Real Classes | Right: K-Means Clusters (Perfect Match!)</em>
</td>
</tr>
</table>

### 📊 Metric Heatmaps (Performance Overview)

<table>
<tr>
<td align="center">
<strong>ARI & NMI Scores across Embeddings</strong><br/>
<img src="results/figures/clustering_metrics_heatmap.png" width="100%"/><br/>
<em>Darker Red = Better Performance. Note how BGE dominates.</em>
</td>
</tr>
</table>

</div>

<br/>

## `> results_summary`

<table align="center">
<tr>
<td width="50%" align="center">
<h3>📊 PT-6 Dataset (Short Texts)</h3>

| Metric | Best Value | Model |
|--------|-------|-------|
| **ARI** | **0.941** | BGE + K-Means |
| **NMI** | **0.935** | BGE + K-Means |
| **Purity** | **0.974** | BGE + GMM |
| **Silhouette** | **0.224** | SBERT + DBSCAN |

<img src="https://img.shields.io/badge/Result-Near_Perfect-00C853?style=flat-square"/>
</td>
<td width="50%" align="center">
<h3>📊 20NG-6 Dataset (News)</h3>

| Metric | Best Value | Model |
|--------|-------|-------|
| **ARI** | **0.60** | BGE + GMM |
| **NMI** | **0.66** | BGE + GMM |
| **Purity** | **0.78** | BGE + K-Means |
| **Silhouette** | **0.15** | TFIDF + DBSCAN |

<img src="https://img.shields.io/badge/Result-Solid_Baseline-4A90E2?style=flat-square"/>
</td>
</tr>
</table>

<br/>

## `> execution`

```bash
# Clone repository
git clone https://github.com/takaokensei/nlp-clustering-benchmark.git
cd nlp-clustering-benchmark

# Create virtual environment (uv or venv)
uv venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline (in order)
# 1. Data Preparation
jupyter notebook notebooks/01_data_prep.ipynb

# 2. Embedding Generation (Cached)
jupyter notebook notebooks/02_embeddings.ipynb

# 3. Clustering (The Main Benchmark)
jupyter notebook notebooks/03_clustering.ipynb

# 4. Visualization
jupyter notebook notebooks/04_visualization.ipynb
```

<br/>

## `> contact`

<div align="center">
  
  <strong>Cauã Vitor Figueredo Silva</strong>
  <br/>
  <samp>Electrical Engineering Student</samp>
  <br/>
  <samp>UFRN - Federal University of Rio Grande do Norte</samp>
  
  <br/><br/>
  
  <a href="https://github.com/takaokensei">
    <img src="https://img.shields.io/badge/-GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/>
  </a>
</div>

<br/>

<div align="center">
  <img src="https://img.shields.io/badge/Made_with-Python_3.12-EE4C2C?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-1a1a2e?style=for-the-badge"/>
</div>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=1a1a2e&height=100&section=footer"/>
