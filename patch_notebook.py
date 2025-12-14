import json
from pathlib import Path
import os

# Define path
nb_path = Path(r"c:\nlp-clustering-benchmark\notebooks\03_clustering.ipynb")

if not nb_path.exists():
    print(f"Error: {nb_path} does not exist")
    exit(1)

print(f"Reading {nb_path}...")
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Helper to find cell
def find_cell_idx(nb, unique_text):
    for idx, cell in enumerate(nb['cells']):
        source_str = "".join(cell['source'])
        if unique_text in source_str:
            return idx
    return -1

# 1. Update Imports
import_idx = find_cell_idx(nb, "from src.config import")
if import_idx >= 0:
    print("Found import cell, patching...")
    source = "".join(nb['cells'][import_idx]['source'])
    
    old_config = "EMBEDDINGS_DIR, CLUSTERING_CONFIGS, N_CLUSTERS, RANDOM_STATE, FIGURES_DIR"
    new_config = "EMBEDDINGS_DIR, CLUSTERING_CONFIGS, N_CLUSTERS, RANDOM_STATE, FIGURES_DIR, PCA_N_COMPONENTS, TIMEOUT_SECONDS"
    
    old_utils = "load_embedding, compute_all_metrics, create_results_dataframe,\n    save_results_table, TABLES_DIR"
    new_utils = "load_embedding, compute_all_metrics, create_results_dataframe,\n    save_results_table, TABLES_DIR, apply_pca, append_result_to_csv, load_checkpoint_results"
    
    source = source.replace(old_config, new_config)
    source = source.replace(old_utils, new_utils)
    
    nb['cells'][import_idx]['source'] = source.splitlines(keepends=True)
else:
    print("Warning: Import cell not found")

# 2. Update Main Loop
loop_idx = find_cell_idx(nb, "all_results = []")
if loop_idx >= 0:
    print("Found loop cell, patching...")
    source = "".join(nb['cells'][loop_idx]['source'])
    
    # Inject Checkpoint
    checkpoint_block = """# Arquivo de checkpoint
CHECKPOINT_FILE = TABLES_DIR / "clustering_results_checkpoint.csv"

# Carregar resultados já processados
existing_results = load_checkpoint_results(CHECKPOINT_FILE)
done_combinations = {(r['dataset'], r['embedding'], r['algorithm']) for r in existing_results}
all_results = existing_results.copy()
print(f"🔄 Recuverados {len(all_results)} resultados do checkpoint.")

# Iterar sobre todas as combinações"""
    
    source = source.replace("# Armazenar todos os resultados\nall_results = []\n\n# Iterar sobre todas as combinações", checkpoint_block)
    
    # Inject PCA and Continue
    old_loop_start = """        for emb_type, X in dataset_data['embeddings'].items():
            for algo_name in algorithms:
                pbar.set_description(f"{dataset_name} | {emb_type} | {algo_name}")
                
                try:"""
    
    new_loop_start = """        for emb_type, X_orig in dataset_data['embeddings'].items():
            # Aplicar PCA se for embedding denso e não for tfidf
            if 'tfidf' not in emb_type:
                X = apply_pca(X_orig, n_components=PCA_N_COMPONENTS)
            else:
                X = X_orig
            
            for algo_name in algorithms:
                # Pular se já feito
                if (dataset_name, emb_type, algo_name) in done_combinations:
                    pbar.update(1)
                    continue

                pbar.set_description(f"{dataset_name} | {emb_type} | {algo_name}")
                
                try:"""
    
    source = source.replace(old_loop_start, new_loop_start)
    
    # Inject Append
    old_append = """                    all_results.append(result)
                    
                except Exception as e:"""
                
    new_append = """                    all_results.append(result)
                    append_result_to_csv(result, CHECKPOINT_FILE)
                    
                except Exception as e:"""
    
    source = source.replace(old_append, new_append)
    
    nb['cells'][loop_idx]['source'] = source.splitlines(keepends=True)
else:
    print("Warning: Loop cell not found")

# Save
print(f"Saving to {nb_path}...")
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print("Done!")
