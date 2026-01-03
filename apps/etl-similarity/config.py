"""
Configurações para os modelos de NLP.

Este arquivo centraliza configurações que podem ser ajustadas sem modificar o código principal.
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Diretório raiz do projeto
PROJECT_ROOT = Path(__file__).parent.parent

# Diretório de assets
# Observação: os assets (stopwords, saídas de similaridade, etc.)
# ficam por padrão em etl-similarity/assets dentro do projeto.
ASSETS_DIR = PROJECT_ROOT / "apps" / "etl-similarity" / "assets"
SHARED_DIR = PROJECT_ROOT / "apps" / "shared"
DATABASE_PATH = PROJECT_ROOT / "database" / "database.db"

# Caminho do modelo FastText
FASTTEXT_MODEL_NAME = "cc.pt.300.bin"

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# Modelo de sentence embeddings
SENTENCE_TRANSFORMER_MODEL = "rufimelo/Legal-BERTimbau-sts-base-ma-v2"

# Modelo de classificação de emoções
EMOTION_MODEL = "pysentimiento/bert-pt-emotion"

# ============================================================================
# PROCESSING PARAMETERS
# ============================================================================

# Parâmetros UMAP para word embeddings
UMAP_WORD_PARAMS = {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "n_components": 2,
    "random_state": 42,
}

# Parâmetros UMAP para sentence embeddings
UMAP_SENT_PARAMS = {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "n_components": 2,
    "random_state": 42,
}

# Número de clusters para KMeans
N_CLUSTERS_WORD = 10
N_CLUSTERS_SENT = 9

# Parâmetros do classificador de emoções
EMOTION_CLASSIFIER_PARAMS = {
    "max_length": 512,
    "truncation": True,
    "device": -1,  # -1 = CPU, 0 = GPU 0, etc.
}

# Número máximo de tokens para análise de emoções
MAX_TOKENS_EMOTION = 400
