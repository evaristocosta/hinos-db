import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from src.loader import hinos_processados, similarity_matrices

hinos: pd.DataFrame = hinos_processados()
similarity_titles, _, _, _ = similarity_matrices()

# separa dados de interesse
hinos["numero"] = hinos.index
hinos_analise = (
    hinos[["numero", "nome", "categoria_abr"]]
    .rename(columns={"numero": "Nº", "nome": "Nome", "categoria_abr": "Categoria"})
    .set_index("Nº")
)
# separa subtitulo do nome
hinos_analise["subtitulo"] = (
    hinos_analise["Nome"].str.extract(r"\((.*?)\)").squeeze().str.strip()
)
hinos_analise["Nome"] = hinos_analise["Nome"].str.replace(
    r"\s*\(.*?\)\s*", "", regex=True
)
# cria dataframe comparativo, considerando o subtitulo como um nome diferente
hinos_titulos = pd.concat(
    [
        hinos_analise[["subtitulo", "Categoria"]].rename(columns={"subtitulo": "Nome"}),
        hinos_analise[["Nome", "Categoria"]],
    ]
).dropna()
# calcula o tamanho do titulo
hinos_analise["titulo_tam_real"] = hinos_analise["Nome"].str.len()
hinos_titulos["titulo_tam_real"] = hinos_titulos["Nome"].str.len()


st.title("🔢 Tamanho dos títulos")

"""
Nesta seção, analisamos o tamanho dos títulos dos hinos na coletânea, tanto considerando
os títulos principais quanto os subtítulos. São considerados subtítulos aqueles que aparecem
entre parênteses no título. 
Na análise com subtítulos, o mesmo hino pode aparecer duas vezes,
uma vez com o título principal e outra com o subtítulo.

O tamanho aqui, é medido em número de caracteres, considerando espaços. 
"""
st.info(
    "É possível usar o filtro na barra lateral para restringir a análise a categorias específicas de hinos.",
    icon="ℹ️",
)


st.sidebar.markdown("# Filtros")
# add filter by category
categorias = hinos_analise["Categoria"].unique()
categoria_selecionada = st.sidebar.multiselect(
    "Filtrar por categoria:", list(categorias), placeholder="Selecione categorias..."
)
if categoria_selecionada:
    hinos_analise_print = hinos_analise.query(
        f"Categoria in {categoria_selecionada}"
    ).copy()
    hinos_titulos = hinos_titulos.query(f"Categoria in {categoria_selecionada}")
else:
    hinos_analise_print = hinos_analise.copy()

col1, col2 = st.columns(2)


with col1:
    st.markdown("**Top 10 maiores títulos**")
    st.dataframe(
        hinos_titulos[["Nome", "titulo_tam_real"]]
        .sort_values(by="titulo_tam_real", ascending=False)
        .head(10),
        column_config={
            "titulo_tam_real": st.column_config.ProgressColumn(
                "Tamanho",
                format="%f",
                help="Tamanho do título em caracteres",
                max_value=int(hinos_titulos["titulo_tam_real"].max()),
                width="small",
            ),
            "Nome": st.column_config.TextColumn(width="small", max_chars=25),
        },
    )


with col2:
    st.markdown("**Top 10 menores títulos**")
    st.dataframe(
        hinos_titulos[["Nome", "titulo_tam_real"]]
        .sort_values(by="titulo_tam_real")
        .head(10),
        column_config={
            "titulo_tam_real": st.column_config.ProgressColumn(
                "Tamanho",
                format="%f",
                help="Tamanho do título em caracteres",
                max_value=int(hinos_titulos["titulo_tam_real"].max()),
                width="small",
            ),
            "Nome": st.column_config.TextColumn(width="small", max_chars=25),
        },
    )


"""
Podemos observar que o maior título contém 46 caracteres, ocorrendo três vezes (hinos 612, 511 e 323).
Já na lista dos menores títulos, vemos que menor título absoluto, com apenas quatro caracteres, é o hino 475 -- Ageu.
"""


"""
## Similaridade entre títulos

Além do tamanho, podemos analisar a similaridade entre os títulos dos hinos.
A seguir, apresentamos uma matriz de similaridade entre os títulos dos hinos, utilizando a métrica de *token set ratio*.
Utilizamos a biblioteca `thefuzz` para calcular a similaridade entre os títulos, que varia de 0 a 100,
onde 100 indica títulos idênticos e 0 indica títulos completamente diferentes. A métrica de *token set ratio*
considera a similaridade entre conjuntos de palavras, ignorando a ordem das palavras e duplicatas.
"""

st.warning(
    "Aplicar filtros pode causar problemas na visualização da matriz de similaridade.",
    icon="⚠️",
)

# restringe a matriz de similaridade aos hinos atualmente no dataframe (caso haja filtro)
idx = hinos_analise.index.tolist()
sim_sub = similarity_titles.loc[idx, idx]

fig = px.imshow(
    sim_sub,
    labels=dict(x="Hinos", y="Hinos", color="Similaridade"),
    x=sim_sub.columns,
    y=sim_sub.index,
    width=600,
    height=600,
    color_continuous_scale="Cividis",
)
st.plotly_chart(fig)

"""
Pela análise da matriz de similaridade, pode-se notar toda sorte de similaridades entre os títulos dos hinos. Por exemplo, 
os hinos 356, 357, 358 e 566 possuem exatamente o mesmo título "O Senhor é o meu pastor", resultando em uma similaridade de 
100 entre eles. Além disso, é interessante notar que existem áreas da matriz onde há uma maior concentração de similaridade, 
indicando grupos de hinos com títulos semelhantes, como é o caso dos hinos de clamor. Não obstante, chama a atenção
algumas linhas e colunas no gráfico que indicam baixa similaridade com todos os outros hinos, sugerindo títulos únicos ou 
muito distintos.
"""

"""
### Hinos com títulos mais similares e menos similares

Para ilustrar melhor as similaridades entre os títulos dos hinos, listamos abaixo todos os pares de hinos com os títulos 
mais similares (>= 80), bem como os 10 pares com os títulos menos similares (< 20).
"""

# pares mais similares

# Obter índices onde a similaridade >= 90 (acima da diagonal)
mask = sim_sub.values >= 90
i_indices, j_indices = np.where(mask)
valid_pairs = i_indices < j_indices  # Manter apenas i < j

similar_pairs = [
    (
        f"{sim_sub.index[i]} - {hinos_analise.loc[sim_sub.index[i], 'Nome']}",
        f"{sim_sub.index[j]} - {hinos_analise.loc[sim_sub.index[j], 'Nome']}",
        sim_sub.iloc[i, j],
    )
    for i, j in zip(i_indices[valid_pairs], j_indices[valid_pairs])
]

similar_pairs_df = pd.DataFrame(
    similar_pairs, columns=["Hino 1", "Hino 2", "Similaridade"]
).sort_values(by="Similaridade", ascending=False)

f"""
#### Pares de hinos com títulos mais similares (>= 90)

Total de pares encontrados: {len(similar_pairs_df)}.
"""

st.dataframe(
    similar_pairs_df,
    hide_index=True,
    column_config={
        "Hino 1": st.column_config.TextColumn(width="small"),
        "Hino 2": st.column_config.TextColumn(width="small"),
        "Similaridade": st.column_config.ProgressColumn(
            "Similaridade",
            format="%f",
            help="Similaridade entre os títulos dos hinos",
            max_value=100,
            width="small",
        ),
    },
)

"""
É possível observar que sem sempre que dois hinos possuam títulos idênticos, eles podem apresentar uma alta similaridade.
Por exemplo, o hino 3 ("Clamo a Ti") tem alta similaridade com qualquer hino que contenha as mesmas palavras, como é o caso dos 
hinos 25, 89 e 295. Isso ocorre porque a métrica de similaridade utilizada considera a presença das palavras,
independentemente da ordem ou de outras palavras adicionais. 

Ainda assim, tal informe pode ser útil para identificar hinos com títulos muito semelhantes, o que pode ser relevante para
seleção durante cultos ou eventos, buscando hinos que estejam de alguma forma correlacionados.

"""

# pares menos similares
dissimilar_pairs = []
mask = (sim_sub.values < 10) & (
    sim_sub.values > 0
)  # Create a mask for dissimilar pairs
i_indices, j_indices = np.where(mask)  # Get indices where the condition is met

for i, j in zip(i_indices, j_indices):
    if i < j:  # Ensure we only take pairs where i < j
        hino_i_nome = hinos_analise.loc[sim_sub.index[i], "Nome"]
        hino_j_nome = hinos_analise.loc[sim_sub.index[j], "Nome"]
        dissimilar_pairs.append(
            (
                f"{sim_sub.index[i]} - {hino_i_nome}",
                f"{sim_sub.index[j]} - {hino_j_nome}",
                sim_sub.iloc[i, j],
            )
        )

dissimilar_pairs_df = pd.DataFrame(
    dissimilar_pairs, columns=["Hino 1", "Hino 2", "Similaridade"]
).sort_values(by="Similaridade")

f"""
#### Pares de hinos com títulos menos similares (> 0, < 10)

Total de pares encontrados: {len(dissimilar_pairs_df)}.
"""

st.dataframe(
    dissimilar_pairs_df,
    hide_index=True,
    column_config={
        "Hino 1": st.column_config.TextColumn(width="small"),
        "Hino 2": st.column_config.TextColumn(width="small"),
        "Similaridade": st.column_config.ProgressColumn(
            "Similaridade",
            format="%f",
            help="Similaridade entre os títulos dos hinos",
            max_value=100,
            width="small",
        ),
    },
)

"""
Me chamou a atenção que hinos que possuem títulos muito específicos, como 396 - "Abba Pai", ou muito curtos e/ou repetitivos, 
como 526 - "Lindo! Lindo! Lindo!", tendem a ter baixa similaridade com outros títulos. Isso sugere que títulos únicos ou
muito distintos podem resultar em menor similaridade, o que é esperado, já que a métrica utilizada valoriza a presença de 
palavras comuns.

"""

"""
### Medidor de título

A seguir, você pode selecionar um hino para ver o tamanho do seu título, comparar com outros hinos com título de 
igual tamanho, e explorar a similaridade do título com os demais hinos da coletânea.
"""

# Criar lista de opções para o selectbox
hinos_opcoes = [f"{num} - {row['Nome']}" for num, row in hinos_analise.iterrows()]

col1, col2 = st.columns(2)

with col1:
    # Selectbox com autocomplete
    hino_selecionado = st.selectbox(
        "Pesquisar hino (número ou nome)",
        options=hinos_opcoes,
        placeholder="Digite para buscar...",
        index=None,
        help="Digite o número ou parte do nome do hino para pesquisar",
    )
    # Extrair o número do hino da seleção
    if hino_selecionado:
        hymn_num = int(hino_selecionado.split(" - ")[0])
        hymn_title = hinos_analise.loc[hymn_num, "Nome"]
        hymn_title_size = hinos_analise.loc[hymn_num, "titulo_tam_real"]

with col2:
    if hino_selecionado:
        st.metric(
            label=f"🎵 Hino {hymn_num} - {hymn_title}",
            value=f"{hymn_title_size} caracteres",
            width="content",
            height="stretch",
        )
    else:
        st.caption("Selecione um hino para ver o tamanho do título.")

if hino_selecionado:
    col1, col2 = st.columns(2)
    with col1:
        hinos_mesmo_tamanho = hinos_analise[
            hinos_analise["titulo_tam_real"] == hymn_title_size
        ].drop(index=hymn_num)
        if not hinos_mesmo_tamanho.empty:
            st.markdown("**Outros hinos com título de igual tamanho:**")

            st.dataframe(
                hinos_mesmo_tamanho[["Nome", "Categoria"]],
                column_config={
                    "Nome": st.column_config.TextColumn(width="small"),
                    "Categoria": st.column_config.TextColumn(
                        width="small", max_chars=25
                    ),
                },
            )
    with col2:
        st.markdown("**Similaridade do título com outros hinos:**")
        sim_title_hymn = similarity_titles.loc[hymn_num].drop(index=hymn_num)
        sim_title_hymn = sim_title_hymn.sort_values(ascending=False)
        # add nome do hino
        sim_title_hymn = sim_title_hymn.to_frame(name="Similaridade")
        sim_title_hymn["Nome"] = sim_title_hymn.index.map(hinos_analise["Nome"])
        sim_title_hymn = sim_title_hymn[["Nome", "Similaridade"]]
        # rename index
        sim_title_hymn.index.name = "Nº"

        st.dataframe(
            sim_title_hymn.head(10),
            column_config={
                "Nome": st.column_config.TextColumn(width="small"),
                "Similaridade": st.column_config.ProgressColumn(
                    "Similaridade",
                    format="%f",
                    help="Similaridade entre os títulos dos hinos",
                    max_value=100,
                    width="small",
                ),
            },
        )

else:
    st.info("Selecione um hino.", icon="ℹ️")
