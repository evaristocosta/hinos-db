import streamlit as st
from src.loader import load_data

hinos = load_data()

# Widgets de filtro
st.sidebar.header("Filtros")  # Deixar os filtros na barra lateral

# Filtros
# st.sidebar.subheader("Filtros:")
num_selecionado = st.sidebar.text_input("Número:")
nome_filtro = st.sidebar.text_input("Título:")
texto_filtro = st.sidebar.text_input("Texto:")

# Aplica quebra de linha
hinos["texto"] = hinos["texto"].str.replace(r"\\n", "\n", regex=True)

# Aplicar os filtros
hinos_filtrado = hinos.copy()
if num_selecionado:
    hinos_filtrado = hinos_filtrado[
        hinos_filtrado["numero"].astype(str).str.contains(num_selecionado)
    ]

if nome_filtro:
    hinos_filtrado = hinos_filtrado[
        hinos_filtrado["nome"].str.contains(nome_filtro, case=False, na=False)
    ]


if texto_filtro:
    hinos_filtrado = hinos_filtrado[
        hinos_filtrado["texto"].str.contains(texto_filtro, case=False, na=False)
    ]

# Filtro por categoria
categorias_unicas = hinos["categoria"].unique()
categorias_selecionadas = st.sidebar.multiselect(
    "Filtrar por categoria:", categorias_unicas, placeholder="Selecione categorias..."
)
if categorias_selecionadas:
    hinos_filtrado = hinos_filtrado[
        hinos_filtrado["categoria"].isin(categorias_selecionadas)
    ]


st.title("📆 Tabela de louvores")

"""
A tabela a seguir representa a base de dados de hinos utilizada para as análises presentes neste projeto. Ela também
pode ser usada para pesquisa de hinos específicos, utilizando os filtros disponíveis na barra lateral.
"""


st.dataframe(
    hinos_filtrado[
        [
            "numero",
            "nome",
            "categoria",
            "texto",
        ]
    ].rename(
        columns={
            "numero": "Nº",
            "nome": "Título",
            "categoria": "Categoria",
            "texto": "Texto",
        }
    ),
    hide_index=True,
    width="stretch",
)
