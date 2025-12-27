from sqlalchemy import create_engine
import pandas as pd


def extract(coletanea_id: int | None = None) -> pd.DataFrame:
    # Cria engine para acessar o banco de dados local (assets/database.db)
    engine = create_engine("sqlite:///../../database/database.db")
    connection = engine.connect()

    # Monta a query para buscar hinos e suas categorias
    where_clause = ""
    if coletanea_id is not None:
        where_clause = f"where coletanea_id = {coletanea_id}"
    
    sql_query = f"""
    select
        numero,
        nome,
        texto,
        texto_limpo,
        categoria_id,
        c.descricao as categoria,
        coletanea_id
    from 
        hino
        left join categoria c on c.id = categoria_id
    {where_clause}
    """

    # Executa a consulta e carrega em um DataFrame
    hinos_analise = pd.read_sql_query(sql_query, connection)

    # Corrige valores nulos e converte número para inteiro
    hinos_analise.loc[hinos_analise["numero"] == "null", "numero"] = 0
    hinos_analise["numero_int"] = hinos_analise["numero"].astype(int)

    # Remove coluna antiga, renomeia e ordena
    hinos_analise = (
        hinos_analise.drop(columns=["numero"])
        .rename(columns={"numero_int": "numero"})
        .sort_values("numero")
    )

    return hinos_analise
