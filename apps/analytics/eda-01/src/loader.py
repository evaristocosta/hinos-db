import streamlit as st
from sqlalchemy import create_engine
import pandas as pd
from pathlib import Path


@st.cache_data
def load_data() -> pd.DataFrame:
    database_path = (
        Path(__file__).parent.parent.parent.parent.parent / "database" / "database.db"
    )
    if not database_path.exists():
        database_path = Path(__file__).parent.parent / "assets" / "database.db"
        if not database_path.exists():
            st.error(
                "Database file not found. Please ensure the database is available in the expected location."
            )
            raise FileNotFoundError(
                f"Database file not found at {database_path}. Please ensure the database is available."
            )

    engine = create_engine(f"sqlite:///{database_path}")

    # Connect to the database
    connection = engine.connect()

    sql_query = """
    select
        h.id,
        h.numero,
        h.nome,
        h.texto,
        h.texto_processado,
        h.categoria_id,
        c.nome as categoria
    from 
        hino h
        left join categoria c on c.id = h.categoria_id
    where
        h.coletanea_id = 1 -- hinos da coletanea padrao
    """

    hinos_analise = pd.read_sql_query(sql_query, connection)
    return hinos_analise


@st.cache_data
def hinos_processados() -> pd.DataFrame:
    pkl_path = Path(__file__).parent.parent / "assets" / "hinos_analise_final.pkl"
    hinos_processados = pd.read_pickle(pkl_path)
    hinos_processados = hinos_processados.query("coletanea_id == 1").drop(
        columns=["coletanea_id"]
    )
    # fix: contracapa
    # hinos_processados.rename(index={795: 0}, inplace=True)
    return hinos_processados


@st.cache_data
def _hinos_processados_legado() -> pd.DataFrame:
    pkl_path = Path(__file__).parent.parent / "assets" / "hinos_analise_emocoes.pkl"
    hinos_processados = pd.read_pickle(pkl_path)

    return hinos_processados


@st.cache_data
def similarity_matrices():
    similarity_titles = pd.read_pickle(
        Path(__file__).parent.parent / "assets" / "similarity_matrix_titles.pkl"
    )

    similarity_word = pd.read_pickle(
        Path(__file__).parent.parent
        / "assets"
        / "similarity_matrix_word_embeddings.pkl"
    )
    similarity_sent = pd.read_pickle(
        Path(__file__).parent.parent
        / "assets"
        / "similarity_matrix_sentence_embeddings.pkl"
    )
    similarity_emocoes = pd.read_pickle(
        Path(__file__).parent.parent / "assets" / "similarity_matrix_emotions.pkl"
    )
    return similarity_titles, similarity_word, similarity_sent, similarity_emocoes


def contracapa_handler(numero: int) -> int:
    match numero:
        case 0:
            return 795
        case 795:
            return 0
        case _:
            return numero
