print("Loading ETL libraries...")

import shutil
import time
from pathlib import Path
from collections import Counter
from functools import lru_cache
from typing import Optional 
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
import fasttext
import fasttext.util
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import umap
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from transformers import pipeline
import torch
from scipy.stats import entropy
from scipy.spatial.distance import euclidean


# ============================================================================
# MODEL LOADERS - Lazy loading com cache para evitar recarregamento
# ============================================================================


@lru_cache(maxsize=1)
def get_stopwords_nltk():
    """Carrega stopwords do NLTK (cached)."""
    try:
        nltk.download("stopwords", quiet=True)
        return nltk.corpus.stopwords.words("portuguese")
    except Exception as e:
        print(f"Aviso: Erro ao carregar stopwords do NLTK: {e}")
        return []


def _get_fasttext_model_path() -> Path:
    """
    Obtém o caminho onde o FastText salva modelos por padrão.

    Returns:
        Path do diretório de cache do FastText.
    """
    return_path = Path.home() / ".fasttext"
    return_path.mkdir(parents=True, exist_ok=True)
    return return_path


@lru_cache(maxsize=1)
def get_fasttext_model(model_path: Optional[str] = None, auto_download: bool = True):
    """
    Carrega modelo FastText para word embeddings (cached).

    Se o modelo não for encontrado e auto_download=True, tenta baixar automaticamente
    usando fasttext.util.download_model() ou carregar de um caminho local.

    Args:
        model_path: Caminho para o modelo .bin. Se None, usa modelo português baixado.
        auto_download: Se True, baixa automaticamente caso não encontre (padrão: True).

    Returns:
        Modelo FastText carregado.

    Raises:
        FileNotFoundError: Se modelo não for encontrado e auto_download=False ou falhar.

    Examples:
        >>> # Uso padrão - baixa modelo português automaticamente
        >>> model = get_fasttext_model()

        >>> # Desabilitar download automático
        >>> model = get_fasttext_model(auto_download=False)

        >>> # Usar modelo de outro local
        >>> model = get_fasttext_model("/caminho/customizado/modelo.bin")
    """

    # Caso 1: Caminho customizado especificado
    if model_path is not None:
        model_path = Path(model_path)
        if model_path.exists():
            print(f"Carregando modelo FastText de {model_path}...")
            return fasttext.load_model(str(model_path))
        elif auto_download:
            print(
                f"Modelo FastText não encontrado em: {model_path}\n"
                f"Fornecido o caminho mas arquivo não existe."
            )
        else:
            raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")

    # Caso 2: Usar modelo português (padrão)
    print("📥 Obtendo modelo FastText português...")

    try:
        # O fasttext.util.download_model baixa automaticamente para ~/.fasttext
        # Se já existe, usa if_exists='ignore' para não baixar novamente
        fasttext.util.download_model("pt", if_exists="ignore")

        # fasttext salva o modelo no diretório onde está sendo executado
        new_model_path = Path(".") / "cc.pt.300.bin"
        # mover modelo para model_path, se definido
        if model_path is not None:
            shutil.move(str(new_model_path), str(model_path))
        else:
            model_path = new_model_path

        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em: {model_path}\n"
                f"Verifique sua conexão de internet ou permissões de escrita em {new_model_path}."
            )

        print(f"Carregando modelo FastText de {model_path}...")
        return fasttext.load_model(str(model_path))

    except Exception as e:
        if auto_download:
            raise FileNotFoundError(
                f"Erro ao obter modelo FastText: {e}\n\n"
                f"Possíveis soluções:\n"
                f"  1. Verifique sua conexão com internet\n"
                f"  2. Verifique permissões de escrita em: {new_model_path}\n"
                f"  3. Use um caminho customizado: get_fasttext_model('/seu/caminho/modelo.bin')"
            )
        else:
            raise


@lru_cache(maxsize=1)
def get_sentence_transformer(
    model_name: str = "rufimelo/Legal-BERTimbau-sts-base-ma-v2",
):
    """
    Carrega modelo SentenceTransformer para sentence embeddings (cached).

    Args:
        model_name: Nome do modelo no Hugging Face Hub.

    Returns:
        Modelo SentenceTransformer carregado.
    """
    print(f"Carregando SentenceTransformer: {model_name}...")
    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def get_emotion_classifier(model_name: str = "pysentimiento/bert-pt-emotion"):
    """
    Carrega pipeline de classificação de emoções (cached).

    Args:
        model_name: Nome do modelo no Hugging Face Hub.

    Returns:
        Pipeline de classificação configurado.
    """
    print(f"Carregando classificador de emoções: {model_name}...")
    return pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        top_k=None,  # Retorna todas as emoções com scores
        max_length=512,  # Limite máximo de tokens do BERT
        truncation=True,  # Trunca textos longos automaticamente
        device=-1,  # Força uso da CPU para evitar problemas de memória
    )


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================


def process_tokens(hinos_input: pd.DataFrame, assets_folder: Path) -> pd.DataFrame:
    """
    Tokeniza os hinos e remove stopwords.

    Args:
        hinos_input: DataFrame com os hinos.
        assets_folder: Pasta com arquivos auxiliares (stopwords-br.txt).

    Returns:
        DataFrame com colunas de tokens adicionadas.
    """
    hinos_analise = hinos_input.copy()

    hinos_analise = hinos_analise.set_index("numero")
    hinos_analise["categoria_abr"] = hinos_analise["categoria"].apply(
        lambda x: x[:13] + "..." if isinstance(x, str) and len(x) > 15 else x
    )

    with open(assets_folder / "stopwords-br.txt", "r", encoding="utf-8") as f:
        stopwords = f.read().splitlines()

    # remover linhas que comecao com #
    stopwords = [eval(word) for word in stopwords if not word.startswith("#")]
    stopwords.extend(["ó", "ti", "pra", "lo", "oh", "és"])

    # merge com stopwords do NLTK
    stopwords_nltk = get_stopwords_nltk()
    stopwords = list(set(stopwords + stopwords_nltk))

    all_tokens = []
    all_tokens_no_stops = []

    for hino in tqdm(hinos_analise.to_dict("records")):
        tokens = nltk.tokenize.regexp_tokenize(hino["texto_limpo"], r"\w+")
        # Replace "MINH" com "MINHA" usando regex
        tokens = [
            nltk.re.sub(r"^minh$", "minha", palavra.lower()) for palavra in tokens
        ]
        tokens_no_stops = [
            palavra for palavra in tokens if palavra.lower() not in stopwords
        ]
        # remover pontuacao
        tokens = [palavra for palavra in tokens if palavra.isalpha()]
        tokens_no_stops = [palavra for palavra in tokens_no_stops if palavra.isalpha()]

        all_tokens.append(tokens)
        all_tokens_no_stops.append(tokens_no_stops)

    hinos_analise["tokens"] = all_tokens
    hinos_analise["tokens_no_stops"] = all_tokens_no_stops
    # considerando numero total de palavras, pois todas elas tem que ser cantadas, logo impactam no tamanho prático do hino
    hinos_analise["num_tokens"] = hinos_analise["tokens"].apply(len)

    # Garantir que 'categoria_id' é tratado como uma variável categórica
    hinos_analise["categoria_id"] = hinos_analise["categoria_id"].astype("category")

    return hinos_analise


def process_ngrams(hinos_input: pd.DataFrame) -> pd.DataFrame:
    """
    Gera bigramas e trigramas dos tokens.

    Args:
        hinos_input: DataFrame com coluna 'tokens_no_stops'.

    Returns:
        DataFrame com colunas de n-gramas adicionadas.
    """
    hinos_analise = hinos_input.copy()

    def get_bigrams(tokens):
        return list(nltk.ngrams(tokens, 2))

    def get_trigrams(tokens):
        return list(nltk.ngrams(tokens, 3))

    # Gerar bigramas para todos os hinos
    hinos_analise["bigrams"] = hinos_analise["tokens_no_stops"].apply(get_bigrams)
    # Gerar trigrams para todos os hinos
    hinos_analise["trigrams"] = hinos_analise["tokens_no_stops"].apply(get_trigrams)
    # Juntar os tokens em strings
    hinos_analise["tokens_str"] = hinos_analise["tokens_no_stops"].apply(
        lambda t: " ".join(t)
    )

    return hinos_analise


def process_word_embeddings(
    hinos_input: pd.DataFrame, model_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Gera word embeddings usando FastText e reduz dimensionalidade com UMAP.

    Args:
        hinos_input: DataFrame com coluna 'tokens_no_stops'.
        model_path: Caminho opcional para modelo FastText customizado.

    Returns:
        DataFrame com embeddings e clusters adicionados.
    """
    hinos_analise = hinos_input.copy()

    # Carrega modelo FastText (lazy loading)
    model_word = get_fasttext_model(model_path)

    def embed_text_weighted(tokens, model):
        if not tokens:
            return np.zeros(model.get_dimension())

        vectors = []
        weights = []

        # Peso baseado em frequência inversa (palavras raras = mais peso)
        token_counts = Counter(tokens)
        total_docs = len(hinos_analise)  # ou seu corpus total

        for word in tokens:
            vector = model.get_word_vector(word)
            # Simulação simples de TF-IDF
            tf = token_counts[word] / len(tokens)
            idf = np.log(
                total_docs
                / (
                    1
                    + sum(
                        1
                        for doc_tokens in hinos_analise["tokens_no_stops"]
                        if word in doc_tokens
                    )
                )
            )
            weight = tf * idf

            vectors.append(vector)
            weights.append(weight)

        # Média ponderada
        weighted_sum = np.average(vectors, axis=0, weights=weights)
        return weighted_sum

    hinos_analise["word_embedding_tfidf"] = hinos_analise["tokens_no_stops"].apply(
        lambda t: embed_text_weighted(t, model_word)
    )

    umap_model = umap.UMAP(
        n_neighbors=15,  # controla quão “local” é o agrupamento (10–50 bons valores)
        min_dist=0.1,  # densidade dos pontos no espaço 2D (0 = pontos bem juntos, 0.5 = mais espalhados)
        n_components=2,  # queremos 2D para visualização
        random_state=42,
    )

    X = np.vstack(hinos_analise["word_embedding_tfidf"].values)
    X_umap = umap_model.fit_transform(X)

    hinos_analise["word_umap1"] = X_umap[:, 0]
    hinos_analise["word_umap2"] = X_umap[:, 1]

    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    hinos_analise["word_cluster"] = kmeans.fit_predict(X_umap)

    n_topics = n_clusters

    # Criar TF-IDF apenas para análise de tópicos
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words=None,  # você já removeu as stopwords
        ngram_range=(1, 3),  # uni, bi e trigramas
        min_df=2,  # palavra deve aparecer em pelo menos 2 documentos
    )

    # Usar texto já limpo (sem stopwords)
    texts_for_topics = [" ".join(tokens) for tokens in hinos_analise["tokens_no_stops"]]
    X_tfidf = vectorizer.fit_transform(texts_for_topics)

    # NMF também funciona com TF-IDF
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=100)
    nmf_topics = nmf.fit_transform(X_tfidf)

    # Atribuir tópicos aos hinos
    hinos_analise["NMF_topic"] = nmf_topics.argmax(axis=1)

    return hinos_analise


def process_sentence_embeddings(
    hinos_input: pd.DataFrame, model_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Gera sentence embeddings usando SentenceTransformer e aplica topic modeling.

    Args:
        hinos_input: DataFrame com coluna 'texto_limpo'.
        model_name: Nome opcional do modelo SentenceTransformer.

    Returns:
        DataFrame com sentence embeddings e tópicos adicionados.
    """
    hinos_analise = hinos_input.copy()

    # Carrega modelo SentenceTransformer (lazy loading)
    model_sent = (
        get_sentence_transformer(model_name)
        if model_name
        else get_sentence_transformer()
    )

    # cria embeddings diretamente para cada hino (texto inteiro)
    print("Computando sentence embeddings...")
    embeddings = model_sent.encode(
        hinos_analise["texto_limpo"].tolist(), show_progress_bar=True
    )
    X_sent = np.array(embeddings)
    hinos_analise["sent_embeddings"] = list(X_sent)

    print("Reduzindo dimensionalidade com UMAP...")
    umap_model = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2, random_state=42
    )
    X_umap = umap_model.fit_transform(X_sent)

    hinos_analise["sent_umap1"] = X_umap[:, 0]
    hinos_analise["sent_umap2"] = X_umap[:, 1]

    print("Clusterizando com KMeans...")
    kmeans = KMeans(n_clusters=9, random_state=42)
    hinos_analise["sent_cluster"] = kmeans.fit_predict(X_umap)

    # Criar o modelo BERTopic
    print("Carregando BERTopic...")
    topic_model = BERTopic(embedding_model=model_sent)

    # Treinar modelo
    print("Treinando BERTopic...")
    topics, _ = topic_model.fit_transform(hinos_analise["texto_limpo"])

    # Associar tópicos ao DataFrame
    hinos_analise["BERT_topic"] = topics

    return hinos_analise


def process_emotions(
    hinos_input: pd.DataFrame, model_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Analisa emoções dos hinos usando classificador BERT.

    Args:
        hinos_input: DataFrame com coluna 'tokens_no_stops'.
        model_name: Nome opcional do modelo de emoções.

    Returns:
        DataFrame com análise de emoções adicionada.
    """
    hinos_analise = hinos_input.copy()

    # Carrega classificador de emoções (lazy loading)
    classifier_emotion = (
        get_emotion_classifier(model_name) if model_name else get_emotion_classifier()
    )

    # Função para juntar e truncar tokens já tratados
    def texto_truncado_tokens(tokens, max_tokens=400):
        if not isinstance(tokens, list):
            return ""
        # Trunca a lista de tokens e junta em uma string
        return " ".join(tokens[:max_tokens])

    # Classificar emoções usando a coluna token_no_stops
    def analisar_emocoes_tokens(tokens):
        try:
            texto = texto_truncado_tokens(tokens)
            if not texto.strip():
                return {}
            resultado = classifier_emotion(texto)
            if isinstance(resultado, list) and len(resultado) > 0:
                emocoes_lista = resultado[0]
                if isinstance(emocoes_lista, list):
                    return {r["label"]: r["score"] for r in emocoes_lista}
            return {}
        except Exception as e:
            print(f"Erro ao processar texto: {str(e)[:100]}...")
            return {}

    # Processar todos os hinos (pode demorar alguns minutos)
    start_time = time.time()

    # Processar em lotes para mostrar progresso
    batch_size = 50
    total_batches = len(hinos_analise) // batch_size + 1

    all_emotions = []
    for i in range(0, len(hinos_analise), batch_size):
        batch = hinos_analise.iloc[i : i + batch_size]
        batch_emotions = batch["tokens_no_stops"].apply(analisar_emocoes_tokens)
        all_emotions.extend(batch_emotions.tolist())

        current_batch = i // batch_size + 1
        print(
            f"Lote {current_batch}/{total_batches} concluído ({i+len(batch)}/{len(hinos_analise)} hinos)"
        )

    # Adicionar resultados ao dataframe
    hinos_analise["emocoes"] = all_emotions

    end_time = time.time()
    print(f"\nProcessamento concluído em {end_time - start_time:.1f} segundos!")
    print(f"Total de hinos processados: {len(hinos_analise)}")

    # ------------------
    # Calculos após obter as emoções
    # ------------------

    # Extrair a emoção dominante de cada hino
    print("Calculando métricas de emoções...")
    emocoes_dominantes = []
    scores_dominantes = []
    emocoes_dominantes_sem_neutral = []
    scores_dominantes_sem_neutral = []

    for emocoes in hinos_analise["emocoes"]:
        if emocoes:
            top_emocao = max(emocoes.items(), key=lambda x: x[1])
            emocoes_dominantes.append(top_emocao[0])
            scores_dominantes.append(top_emocao[1])

            # Remove 'neutral' se existir
            emocoes_filtrado = {k: v for k, v in emocoes.items() if k != "neutral"}
            if emocoes_filtrado:
                top_emocao = max(emocoes_filtrado.items(), key=lambda x: x[1])
                emocoes_dominantes_sem_neutral.append(top_emocao[0])
                scores_dominantes_sem_neutral.append(top_emocao[1])
            else:
                emocoes_dominantes_sem_neutral.append("unknown")
                scores_dominantes_sem_neutral.append(0.0)
        else:
            emocoes_dominantes.append("unknown")
            scores_dominantes.append(0.0)

            emocoes_dominantes_sem_neutral.append("unknown")
            scores_dominantes_sem_neutral.append(0.0)

    hinos_analise["emocao_dominante"] = emocoes_dominantes
    hinos_analise["score_dominante"] = scores_dominantes
    hinos_analise["emocao_dominante_sem_neutral"] = emocoes_dominantes_sem_neutral
    hinos_analise["score_dominante_sem_neutral"] = scores_dominantes_sem_neutral

    print("Calculando diversidade e concentração emocional...")
    def calcular_diversidade_emocional(emocoes):
        """Calcula a entropia de Shannon para medir diversidade emocional"""
        if not emocoes:
            return 0.0
        scores = np.array(list(emocoes.values()))
        # Normalizar para que somem 1 (distribuição de probabilidade)
        if scores.sum() > 0:
            probs = scores / scores.sum()
            return entropy(probs)
        return 0.0

    def calcular_concentracao_emocional(emocoes):
        """Calcula índice de concentração (Gini simplificado)"""
        if not emocoes:
            return 0.0
        scores = np.array(list(emocoes.values()))
        if scores.sum() == 0:
            return 0.0
        # Score máximo / soma total (quanto maior, mais concentrado)
        return scores.max() / scores.sum()

    # Aplicar métricas
    hinos_analise["diversidade_emocional"] = hinos_analise["emocoes"].apply(
        calcular_diversidade_emocional
    )
    hinos_analise["concentracao_emocional"] = hinos_analise["emocoes"].apply(
        calcular_concentracao_emocional
    )

    # Usar índice como proxy
    hinos_analise["posicao_percentil"] = pd.qcut(
        hinos_analise.index,
        q=4,
        labels=["Início (25%)", "Meio-Início (50%)", "Meio-Fim (75%)", "Fim (100%)"],
        duplicates="drop",
    )

    print("Calculando scores líquidos e categorias emocionais...")
    def calcular_score_liquido(emocoes):
        """Calcula a diferença entre emoção dominante (não-neutral) e neutral"""
        if not emocoes:
            return 0.0, "unknown"

        score_neutral = emocoes.get("neutral", 0.0)
        emocoes_sem_neutral = {k: v for k, v in emocoes.items() if k != "neutral"}

        if emocoes_sem_neutral:
            top_emocao, top_score = max(emocoes_sem_neutral.items(), key=lambda x: x[1])
            return top_score - score_neutral, top_emocao
        return 0.0, "unknown"

    # Aplicar cálculo
    scores_liquidos = hinos_analise["emocoes"].apply(calcular_score_liquido)
    hinos_analise["score_liquido"] = scores_liquidos.apply(lambda x: x[0])
    hinos_analise["emocao_dominante_nao_neutral"] = scores_liquidos.apply(
        lambda x: x[1]
    )

    bins = [-1, -0.1, 0.1, 0.3, 1]
    labels = ["Muito Neutro", "Neutro", "Emocional", "Muito Emocional"]
    hinos_analise["categoria_emocional"] = pd.cut(
        hinos_analise["score_liquido"], bins=bins, labels=labels
    )

    CATEGORIAS_EMOCOES = {
        "positivas": [
            "joy",
            "love",
            "admiration",
            "approval",
            "caring",
            "excitement",
            "gratitude",
            "optimism",
            "pride",
            "relief",
        ],
        "negativas": [
            "anger",
            "disgust",
            "fear",
            "sadness",
            "disappointment",
            "embarrassment",
            "grief",
            "nervousness",
            "remorse",
        ],
        "neutras": [
            "neutral",
            "realization",
            "surprise",
            "confusion",
            "curiosity",
            "desire",
            "amusement",
        ],
    }

    def calcular_score_categoria(emocoes, categoria):
        """Soma os scores de todas as emoções de uma categoria"""
        if not emocoes:
            return 0.0
        emocoes_da_categoria = CATEGORIAS_EMOCOES.get(categoria, [])
        return sum(emocoes.get(emocao, 0.0) for emocao in emocoes_da_categoria)

    # Aplicar para cada categoria
    for categoria in ["positivas", "negativas", "neutras"]:
        hinos_analise[f"score_{categoria}"] = hinos_analise["emocoes"].apply(
            lambda x: calcular_score_categoria(x, categoria)
        )

    # Categoria dominante
    def categoria_dominante(row):
        scores = {
            "positivas": row["score_positivas"],
            "negativas": row["score_negativas"],
            "neutras": row["score_neutras"],
        }
        return max(scores.items(), key=lambda x: x[1])[0]

    hinos_analise["categoria_dominante"] = hinos_analise.apply(
        categoria_dominante, axis=1
    )

    # Criar vetor de emoções médias
    print("Calculando métricas adicionais de emoções...")
    emocoes_todas = set()
    for emocoes in hinos_analise["emocoes"]:
        if emocoes:
            emocoes_todas.update(emocoes.keys())

    vetor_medio = {}
    for emocao in emocoes_todas:
        scores = [e.get(emocao, 0.0) for e in hinos_analise["emocoes"] if e]
        vetor_medio[emocao] = np.mean(scores) if scores else 0.0

    def calcular_distancia_media(emocoes):
        if not emocoes:
            return 0.0
        vetor_hino = [emocoes.get(emocao, 0.0) for emocao in sorted(vetor_medio.keys())]
        vetor_medio_sorted = [
            vetor_medio[emocao] for emocao in sorted(vetor_medio.keys())
        ]
        return euclidean(vetor_hino, vetor_medio_sorted)

    hinos_analise["distancia_perfil_medio"] = hinos_analise["emocoes"].apply(
        calcular_distancia_media
    )

    # Intensidade emocional total (soma das emoções não-neutras)
    def calcular_intensidade_emocional(emocoes):
        """Soma dos scores de todas as emoções exceto neutral"""
        if not emocoes:
            return 0.0
        return sum(v for k, v in emocoes.items() if k != "neutral")

    hinos_analise["intensidade_emocional"] = hinos_analise["emocoes"].apply(
        calcular_intensidade_emocional
    )

    # Complexidade emocional (número de emoções acima de um threshold)
    def calcular_num_emocoes_fortes(emocoes, threshold=0.1):
        """Conta quantas emoções têm score acima do threshold"""
        if not emocoes:
            return 0
        return sum(1 for k, v in emocoes.items() if v >= threshold and k != "neutral")

    hinos_analise["num_emocoes_fortes"] = hinos_analise["emocoes"].apply(
        calcular_num_emocoes_fortes
    )

    # Score de "Alegria Líquida" (joy - sadness)
    def calcular_alegria_liquida(emocoes):
        if not emocoes:
            return 0.0
        joy = emocoes.get("joy", 0.0)
        sadness = emocoes.get("sadness", 0.0)
        return joy - sadness

    hinos_analise["alegria_liquida"] = hinos_analise["emocoes"].apply(
        calcular_alegria_liquida
    )

    return hinos_analise
