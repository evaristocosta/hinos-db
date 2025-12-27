import pathlib
import pandas as pd
import numpy as np
from tqdm import tqdm
from thefuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def similarity_title(hinos_analise: pd.DataFrame, assets_folder: pathlib.Path) -> None:
    # Extrai subtítulos entre parênteses e limpa a coluna "nome"
    hinos_analise["subtitulo"] = (
        hinos_analise["nome"].str.extract(r"\((.*?)\)").squeeze().str.strip()
    )
    hinos_analise["nome"] = hinos_analise["nome"].str.replace(
        r"\s*\(.*?\)\s*", "", regex=True
    )

    hinos_titulos = pd.concat(
        [
            hinos_analise[["subtitulo", "categoria"]].rename(
                columns={"subtitulo": "nome"}
            ),
            hinos_analise[["nome", "categoria"]],
        ]
    ).dropna()
    hinos_titulos["titulo_tam_real"] = hinos_titulos["nome"].str.len()

    indices_unicos = sorted(hinos_titulos.index.unique())
    n = len(indices_unicos)

    # Cria matriz de similaridade usando token_set_ratio
    # Para hinos com múltiplos títulos (título + subtítulo), usa o maior valor
    matriz_similaridade = np.zeros((n, n))

    for i, idx1 in tqdm(enumerate(indices_unicos), total=n):
        for j, idx2 in enumerate(indices_unicos):
            if i == j:
                matriz_similaridade[i, j] = 100  # Similaridade consigo mesmo
            elif i < j:  # Calcula apenas metade para otimizar
                # Pega todos os títulos do hino 1 e do hino 2
                titulos_h1 = hinos_titulos.loc[idx1, "nome"]
                titulos_h2 = hinos_titulos.loc[idx2, "nome"]

                # Garante que sejam listas
                if isinstance(titulos_h1, str):
                    titulos_h1 = [titulos_h1]
                else:
                    titulos_h1 = titulos_h1.tolist()

                if isinstance(titulos_h2, str):
                    titulos_h2 = [titulos_h2]
                else:
                    titulos_h2 = titulos_h2.tolist()

                # Calcula similaridade entre todos os pares e pega o máximo
                max_sim = 0
                for t1 in titulos_h1:
                    for t2 in titulos_h2:
                        sim = fuzz.token_set_ratio(t1, t2)
                        max_sim = max(max_sim, sim)

                matriz_similaridade[i, j] = max_sim
                matriz_similaridade[j, i] = max_sim  # Matriz simétrica

    # Cria DataFrame com a matriz
    df_matriz = pd.DataFrame(
        matriz_similaridade, index=indices_unicos, columns=indices_unicos
    )

    df_matriz.to_pickle(assets_folder / "similarity_matrix_titles.pkl")


def similarity_word(hinos_analise: pd.DataFrame, assets_folder: pathlib.Path) -> None:
    # TF-IDF: unigrams e bigrams
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=None)
    X_tfidf = vectorizer.fit_transform(hinos_analise["tokens_str"])

    similarity_tfidf = cosine_similarity(X_tfidf)
    similarity_df_tfidf = pd.DataFrame(
        similarity_tfidf, index=hinos_analise.index, columns=hinos_analise.index
    )

    similarity_df_tfidf.to_pickle(assets_folder / "similarity_matrix_words.pkl")


def similarity_word_embeddings(
    hinos_analise: pd.DataFrame, assets_folder: pathlib.Path
) -> None:
    sims_tfidf = cosine_similarity(list(hinos_analise["word_embedding_tfidf"]))

    # hinos mais semelhantes ao hino 443
    similarities_tfidf = list(enumerate(sims_tfidf[443]))
    similarities_tfidf = sorted(similarities_tfidf, key=lambda x: x[1], reverse=True)

    sims_tfidf_df = pd.DataFrame(
        sims_tfidf, index=hinos_analise.index, columns=hinos_analise.index
    )

    sims_tfidf_df.to_pickle(assets_folder / "similarity_matrix_word_embeddings.pkl")


def similarity_sentence_embeddings(
    hinos_analise: pd.DataFrame, assets_folder: pathlib.Path
) -> None:
    similarity_matrix = cosine_similarity(list(hinos_analise["sent_embeddings"]))

    sims_sent_df = pd.DataFrame(
        similarity_matrix, index=hinos_analise.index, columns=hinos_analise.index
    )

    sims_sent_df.to_pickle(assets_folder / "similarity_matrix_sentence_embeddings.pkl")


def similarity_emotions(
    hinos_analise: pd.DataFrame, assets_folder: pathlib.Path
) -> None:
    # Preparar matriz de features (cada linha é um hino, colunas são as emoções)
    # Primeiro, identificar todas as emoções presentes
    todas_emocoes = sorted(
        set(
            emocao
            for emocoes_dict in hinos_analise["emocoes"]
            for emocao in emocoes_dict.keys()
        )
    )

    # Criar matriz de features
    matriz_features = np.array(
        [
            [hino["emocoes"].get(emocao, 0.0) for emocao in todas_emocoes]
            for _, hino in hinos_analise.iterrows()
        ]
    )

    # Calcular similaridade de cosseno (uma única chamada!)
    matriz_similaridade = cosine_similarity(matriz_features)

    # Criar DataFrame com os índices
    matriz_sim_df = pd.DataFrame(
        matriz_similaridade, index=hinos_analise.index, columns=hinos_analise.index
    )

    matriz_sim_df.to_pickle(assets_folder / "similarity_matrix_emotions.pkl")
