import streamlit as st
import altair as alt
from src.loader import hinos_processados

hinos = hinos_processados()
hinos["numero"] = hinos.index

categorias_count = (
    hinos[["categoria_abr", "numero"]]
    .groupby(["categoria_abr"])
    .count()
    .reset_index()
    .rename(columns={"numero": "contagem"})
)

# calcular porcentagem e string formatada para tooltip
total = categorias_count["contagem"].sum()
if total > 0:
    categorias_count["porcentagem"] = categorias_count["contagem"] / total * 100
else:
    categorias_count["porcentagem"] = 0
categorias_count["porcentagem_str"] = categorias_count["porcentagem"].map(
    lambda v: f"{v:.2f}%"
)

# marcar barras maior / menor para color mapping
if not categorias_count.empty:
    max_val = categorias_count["contagem"].max()
    min_val = categorias_count["contagem"].min()

    def bar_type(v):
        if v == max_val:
            return "Maior"
        if v == min_val:
            return "Menor"
        return "Outro"

    categorias_count["bar_type"] = categorias_count["contagem"].apply(bar_type)
else:
    categorias_count["bar_type"] = []

st.title("📑 Categoria dos louvores")

"""
Nesta seção, exploramos a distribuição de hinos por categoria da coletânea.
Tais categorias são definidas pela própria coletânea.
"""


sort_by = st.selectbox(
    "Ordenar por:",
    options=[
        "Nome da categoria",
        "Número de hinos (ascendente)",
        "Número de hinos (descendente)",
    ],
    index=0,
)

if sort_by == "Nome da categoria":
    categorias_count = categorias_count.sort_values("categoria_abr").reset_index(
        drop=True
    )
    y_order = categorias_count["categoria_abr"].tolist()
elif sort_by == "Número de hinos (ascendente)":
    categorias_count = categorias_count.sort_values("contagem").reset_index(drop=True)
    y_order = categorias_count["categoria_abr"].tolist()
elif sort_by == "Número de hinos (descendente)":
    categorias_count = categorias_count.sort_values(
        "contagem", ascending=False
    ).reset_index(drop=True)
    y_order = categorias_count["categoria_abr"].tolist()

# exibir grafico de barras ordenadas de acordo com opcao (Altair para controle de tooltip/ordenacao)
chart = (
    alt.Chart(categorias_count)
    .mark_bar()
    .encode(
        x=alt.X("contagem:Q", title="Número de hinos"),
        y=alt.Y("categoria_abr:N", title="Categoria", sort=y_order),
        tooltip=[
            alt.Tooltip("categoria_abr:N", title="Categoria"),
            alt.Tooltip("contagem:Q", title="Número de hinos"),
            alt.Tooltip("porcentagem_str:N", title="Porcentagem"),
        ],
        color=alt.Color(
            "bar_type:N",
            scale=alt.Scale(
                domain=["Maior", "Menor", "Outro"],
                range=["#818d3f", "#d7a04f", "#6181a8"],
            ),
            legend=None,
        ),
    )
    .properties(height=500)
)

st.altair_chart(chart, width="stretch")


# conclusões
"""
A categoria "DEDICAÇÃO" é a maior em quantidade, com 104 hinos (13,08%), sendo a categoria mais representada.
Outras categorias bastante presentes são "VOLTA DE JESUS E ETERNIDADE" e "MORTE, RESSURREIÇÃO E SALVAÇÃO", 
ambas com 94 hinos cada (11,82% cada), seguidas de "SANTIFICAÇÃO E DERRAMAMENTO DO ESPÍRITO SANTO" 
(92 hinos, 11,57%), "CONSOLO E ENCORAJAMENTO" (91 hinos, 11,45%) e "LOUVOR" (78 hinos, 9,81%).
Categorias pequenas incluem "SALMOS DE LOUVOR" com apenas 16 hinos (2,01%), destacando-se como a menos representada.
Por fim, a distribuição geral não é uniforme: há predominância de categorias relacionados a dedicação, 
volta de Jesus, santificação e consolo, enquanto Salmos de louvor e invocação são menos frequentes.
"""
