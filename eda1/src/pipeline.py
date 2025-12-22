import streamlit as st
from sqlalchemy import create_engine
import pandas as pd
from pathlib import Path

# import nltk

# nltk.download("stopwords")


@st.cache_data
def hinos_processados() -> pd.DataFrame:
    # hinos = load_data()
    # hinos = preprocessing(hinos)
    pkl_path = Path(__file__).parent.parent / "assets" / "hinos_analise_emocoes.pkl"
    hinos_processados = pd.read_pickle(pkl_path)

    return hinos_processados


@st.cache_data
def load_data() -> pd.DataFrame:
    database_path = Path(__file__).parent.parent / "assets" / "database.db"
    engine = create_engine(f"sqlite:///{database_path}")

    # Connect to the database
    connection = engine.connect()

    sql_query = """
    select
        numero,
        nome,
        texto,
        texto_limpo,
        categoria_id,
        c.descricao as categoria
    from 
        hino
        left join categoria c on c.id = categoria_id
    where
        coletanea_id = 1
    """

    hinos_analise = pd.read_sql_query(sql_query, connection)
    return hinos_analise


@st.cache_data
def similarity_matrices():
    similarity_titles = pd.read_pickle(
        Path(__file__).parent.parent / "assets" / "similarity_matrix_titles.pkl"
    )

    similarity_word = pd.read_pickle(
        Path(__file__).parent.parent
        / "assets"
        / "similarity_matrix_word_embeddings_tfidf.pkl"
    )
    similarity_sent = pd.read_pickle(
        Path(__file__).parent.parent
        / "assets"
        / "similarity_matrix_sentence_embeddings.pkl"
    )
    similarity_emocoes = pd.read_pickle(
        Path(__file__).parent.parent / "assets" / "similarity_matrix_emocoes.pkl"
    )
    return similarity_titles, similarity_word, similarity_sent, similarity_emocoes
