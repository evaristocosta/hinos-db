import streamlit as st
import pandas as pd
from src.loader import hinos_processados, similarity_matrices
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from scipy.spatial.distance import euclidean

# Emoções (eda1_part7):
st.title("🎭 Análise de emoções nos hinos")

hinos_analise: pd.DataFrame = hinos_processados()
_, _, _, similarity_emocoes = similarity_matrices()

# modelo usado: https://huggingface.co/pysentimiento/bert-pt-emotion
"""
Nesta página, exploramos as emoções expressas nos hinos utilizando análises quantitativas e 
visuais. As emoções foram classificadas usando o modelo [bert-pt-emotion](https://huggingface.co/pysentimiento/bert-pt-emotion), 
um modelo de  processamento de linguagem natural treinado para reconhecer emoções em textos 
em português. As principais emoções analisadas incluem alegria, tristeza, otimismo, luto,
entre outras.
"""

# top 10 emoções dominantes nos hinos (sem 'neutral')

"""
## Distribuição das emoções dominantes

No gráfico abaixo, visualizamos as 10 emoções dominantes mais frequentes nos hinos, 
excluindo a categoria 'neutral' (neutra). Isso nos ajuda a entender quais emoções são mais 
prevalentes na coletânea.
"""

st.info(
    """
    O nome das emoções está em inglês por padrão, por conta do modelo utilizado.
""",
    icon="ℹ️",
)

emocao_counts = (
    pd.Series(hinos_analise["emocao_dominante_sem_neutral"].tolist())
    .value_counts()
    .head(10)
)

# Calcular porcentagens
total_hinos = len(hinos_analise)
emocao_df = pd.DataFrame(
    {
        "emocao": emocao_counts.index,
        "contagem": emocao_counts.values,
        "percentual": (emocao_counts.values / total_hinos * 100).round(2),
    }
)

fig_bar = px.bar(
    emocao_df,
    x="contagem",
    y="emocao",
    orientation="h",
    labels={"contagem": "Número de hinos", "emocao": "Emoção"},
    custom_data=["percentual"],
    color_discrete_sequence=["#6181a8"],
)

fig_bar.update_traces(
    hovertemplate="<b>%{y}</b><br>Contagem: %{x}<br>Percentual: %{customdata[0]:.2f}%<extra></extra>"
)

fig_bar.update_layout(height=400, yaxis={"categoryorder": "total ascending"})

st.plotly_chart(fig_bar, width="stretch")

"""
Quase metade dos hinos expressam "amor", e quase um quarto expressam "otimismo", sendo essas
as emoções mais marcantes nos hinos analisados. Além disso, as oito primeiras emoções são positivas,
indicando uma tendência geral de otimismo e alegria na coletânea.
"""

# matriz de correlação
"""
### Matriz de correlação entre emoções

Como análise continuada, exploramos a correlação entre diferentes emoções expressas nos hinos. 
A matriz de correlação abaixo mostra como as emoções se relacionam entre si, indicando quais
emoções tendem a aparecer juntas nos hinos.
"""

emocoes_por_hino = []
for emocoes in hinos_analise["emocoes"]:
    if emocoes:
        emocoes_por_hino.append(emocoes)
    else:
        emocoes_por_hino.append({})

emo_df_completo = pd.DataFrame(emocoes_por_hino).fillna(0)

# Matriz de correlação entre emoções
correlacao_emocoes = emo_df_completo.corr()

# Visualizar matriz de correlação com Plotly

# Criar máscara para triângulo superior
mask = np.triu(np.ones_like(correlacao_emocoes, dtype=bool))
correlacao_masked = correlacao_emocoes.copy()
correlacao_masked = correlacao_masked.where(~mask)

# Criar matriz de texto, substituindo NaN por strings vazias
texto_correlacao = correlacao_masked.round(2).astype(str)
texto_correlacao = texto_correlacao.replace("nan", "")

fig_corr = px.imshow(
    correlacao_masked,
    color_continuous_scale="RdBu_r",
    color_continuous_midpoint=0,
    aspect="auto",
    labels=dict(color="Correlação"),
    # title="Matriz de Correlação entre Emoções<br>(Como as emoções se relacionam entre si)"
)
fig_corr.update_traces(text=texto_correlacao, texttemplate="%{text}")
fig_corr.update_layout(height=700)
st.plotly_chart(fig_corr, width="stretch")

"""
Quatro correlações notáveis emergem da análise: tristeza ("sadness") e luto ("grief") estão 
fortemente correlacionados -- a maior correlação observada -- sugerindo que hinos que 
expressam tristeza frequentemente também abordam temas de perda e luto. Segundo, confusão ("confusion")
e curiosidade ("curiosity") mostram uma correlação positiva significativa, seguido de
medo ("fear") com nervosismo ("nervousness"), e alegria ("joy") e alívio ("relief").

É possível notar algumas linhas também, indicando que certas emoções têm correlações mais fortes,
ou inversas entre si. É o caso de embaraço ("embarrassment"), que tem correlações positivas
com várias emoções negativas. Por outro lado, a emoção neutra ("neutral") mostra correlações
inversas com várias emoções, principalmente amor ("love") -- a emoção mais presente
nos hinos -- indicando que hinos neutros tendem a evitar expressar emoções fortes.
"""

# diversidade (shannon, concentração, exemplos)
"""
## Diversidade emocional nos hinos

A diversidade emocional nos hinos é medida pela Entropia de Shannon, que captura a variedade e a distribuição das 
emoções expressas. Quanto maior a entropia, maior a diversidade emocional. Além disso, analisamos a concentração emocional, 
que indica o quão dominante é a emoção principal em relação ao total. Uma alta concentração sugere que um hino é fortemente 
dominado por uma única emoção, enquanto uma baixa concentração indica um equilíbrio entre múltiplas emoções.
"""

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        "Distribuição da Diversidade Emocional<br>(Entropia de Shannon)",
        "Distribuição da Concentração Emocional<br>(Score máximo / Total)",
    ),
)

# Histograma de diversidade
fig.add_trace(
    go.Histogram(
        x=hinos_analise["diversidade_emocional"],
        nbinsx=30,
        marker_color="#6181a8",
        opacity=0.7,
        name="Diversidade",
    ),
    row=1,
    col=1,
)
# Linha vertical da média
media_div = hinos_analise["diversidade_emocional"].mean()
fig.add_vline(
    x=media_div,
    line_dash="dash",
    line_color="#d80d11",
    annotation_text=f"Média: {media_div:.3f}",
    row=1,
    col=1,
)

# Histograma de concentração
fig.add_trace(
    go.Histogram(
        x=hinos_analise["concentracao_emocional"],
        nbinsx=30,
        marker_color="#d7a04f",
        opacity=0.7,
        name="Concentração",
    ),
    row=1,
    col=2,
)
# Linha vertical da média
media_conc = hinos_analise["concentracao_emocional"].mean()
fig.add_vline(
    x=media_conc,
    line_dash="dash",
    line_color="#d80d11",
    annotation_text=f"Média: {media_conc:.3f}",
    row=1,
    col=2,
)

fig.update_xaxes(title_text="Entropia", row=1, col=1)
fig.update_xaxes(title_text="Índice de Concentração", row=1, col=2)
fig.update_yaxes(title_text="Frequência", row=1, col=1)
fig.update_yaxes(title_text="Frequência", row=1, col=2)
fig.update_layout(height=500, showlegend=False)

st.plotly_chart(fig, width="stretch")


f"""
A entropia parece ter uma distribuição aproximadamente normal, com a maioria dos hinos
apresentando uma diversidade emocional moderada. A média de entropia é de {hinos_analise['diversidade_emocional'].mean():.3f},
indicando que os hinos tendem a expressar uma variedade razoável de emoções.

Já a concentração emocional mostra uma leve inclinação para valores mais altos,
sugerindo que muitos hinos são dominados por uma ou poucas emoções principais. A média de concentração é de {hinos_analise['concentracao_emocional'].mean():.3f},
ou seja, em média, a emoção dominante representa cerca de {hinos_analise['concentracao_emocional'].mean() * 100:.1f}% do total emocional do hino.
"""


# Exemplos de hinos mais diversos vs. mais concentrados
"""
### Exemplos de hinos com diferentes perfis emocionais

Aqui, destacamos exemplos de hinos que exemplificam diversidade e concentração emocional.

"""


col1, col2 = st.columns(2)

with col1:
    st.write("**Hinos Mais DIVERSOS Emocionalmente (múltiplas emoções balanceadas)**")

    mais_diversos = hinos_analise.nlargest(5, "diversidade_emocional")
    rows_div = []
    for i, (idx, hino) in enumerate(mais_diversos.iterrows(), 1):
        top_3 = ""
        if hino["emocoes"]:
            top_3_list = sorted(
                hino["emocoes"].items(), key=lambda x: x[1], reverse=True
            )[:3]
            top_3 = ", ".join([f"{e[0]}({e[1]:.2f})" for e in top_3_list])
        rows_div.append(
            {
                "rank": i,
                "nome": f"{idx} - {hino['nome']}",
                "entropia": round(hino["diversidade_emocional"], 3),
                "top_3_emocoes": top_3,
            }
        )

    df_div = pd.DataFrame(rows_div).set_index("rank")
    st.dataframe(
        df_div[["nome", "entropia", "top_3_emocoes"]],
        width="stretch",
        column_config={
            "top_3_emocoes": st.column_config.TextColumn(
                "Top 3 Emoções",
                help="As três emoções mais fortes no hino, com seus scores.",
            ),
            "entropia": st.column_config.NumberColumn(
                "Entropia", help="Medida da diversidade emocional no hino."
            ),
            "nome": st.column_config.TextColumn(
                "Nome do Hino", help="Identificação do hino pelo seu índice e nome."
            ),
        },
    )

with col2:
    st.write("**Hinos Mais CONCENTRADOS Emocionalmente (emoção dominante forte)**")

    mais_concentrados = hinos_analise.nlargest(5, "concentracao_emocional")
    rows_conc = []
    for i, (idx, hino) in enumerate(mais_concentrados.iterrows(), 1):
        top_emocao = ""
        if hino["emocoes"]:
            top_emocao_item = max(hino["emocoes"].items(), key=lambda x: x[1])
            top_emocao = f"{top_emocao_item[0]} ({top_emocao_item[1]:.3f})"
        rows_conc.append(
            {
                "rank": i,
                "nome": f"{idx} - {hino['nome']}",
                "concentracao": round(hino["concentracao_emocional"], 3),
                "emocao_dominante": top_emocao,
            }
        )

    df_conc = pd.DataFrame(rows_conc).set_index("rank")
    st.dataframe(
        df_conc[["nome", "concentracao", "emocao_dominante"]],
        width="stretch",
        column_config={
            "emocao_dominante": st.column_config.TextColumn(
                "Emoção Dominante", help="A emoção mais forte no hino, com seu score."
            ),
            "concentracao": st.column_config.NumberColumn(
                "Índice de Concentração",
                help="Medida da concentração emocional no hino.",
            ),
            "nome": st.column_config.TextColumn(
                "Nome do Hino", help="Identificação do hino pelo seu índice e nome."
            ),
        },
    )

"""
Na diversidade, pode-se notar as diferentes emoções que aparecem com scores relativamente próximos: tristeza com prazer, ou
tristeza com amor, etc. Já na concentração, os hinos são todos fortemente dominados pela "falta de emoção" (neutral),
indicando que esses hinos são mais neutros em termos emocionais.
"""

# distribuição de categorias emocionais
"""
## Distribuição das categorias emocionais nos hinos

Das diversas categorias emocionais atribuídas aos hinos, podemos organizá-las em três grandes grupos:

- **Positivas:** categorias que expressam emoções alegres, otimistas e de esperança.
- **Neutras:** categorias que refletem emoções mais contidas ou ambivalentes.
- **Negativas:** categorias que transmitem emoções tristes, pessimistas e de desespero.

A seguir, exploramos a distribuição dessas categorias emocionais na coletânea de hinos.

"""
categoria_counts = hinos_analise["categoria_dominante"].value_counts()

# Gráfico de barras horizontal empilhado das categorias emocionais
color_seq = ["#a3b350", "lightgray", "#d80d11"]
total = int(categoria_counts.sum()) if not np.isnan(categoria_counts.sum()) else 0

fig = go.Figure()
for i, cat in enumerate(categoria_counts.index):
    val = int(categoria_counts.loc[cat])
    pct = (val / total * 100) if total else 0.0
    fig.add_trace(
        go.Bar(
            x=[val],
            y=["Categorias"],
            name=str(cat),
            orientation="h",
            marker_color=color_seq[i % len(color_seq)],
            text=f"{val} ({pct:.1f}%)",
            textposition="inside",
            hovertemplate=f"{cat}: {val} hinos<br>{pct:.1f}%<extra></extra>",
        )
    )

fig.update_layout(
    # barmode='stack',
    # height=300,
    xaxis_title="Número de hinos",
    yaxis={"visible": False},
    legend_title_text="Categoria",
    title="Distribuição das Categorias Emocionais Dominantes nos Hinos",
)

st.plotly_chart(fig, width="stretch")

"""
Pode-se observar que a maioria dos hinos (62,3%) pertence a categorias emocionais positivas, enquanto 33,2% são neutras e 
apenas 4,5% são negativas. Isso reflete uma tendência geral de otimismo e esperança na coletânea, com poucos hinos 
expressando emoções negativas.
"""

# Relação entre positivas e negativas
df_scatter = hinos_analise.copy()
df_scatter["idx"] = df_scatter.index.astype(str)

fig_scatter = px.scatter(
    df_scatter,
    x="score_positivas",
    y="score_negativas",
    color="score_neutras",
    color_continuous_scale="cividis",
    opacity=0.5,
    title="Relação entre Emoções Positivas e Negativas",
    labels={
        "score_positivas": "Score Emoções Positivas",
        "score_negativas": "Score Emoções Negativas",
        "score_neutras": "Score Neutro",
    },
    hover_data={
        "idx": True,
        "nome": True,
        "score_positivas": ":.3f",
        "score_negativas": ":.3f",
        "score_neutras": ":.3f",
    },
)
fig_scatter.update_layout(height=500)
st.plotly_chart(fig_scatter, width="stretch")

"""
A visualização de relacionamento deixa ainda mais clara a positividade predominante nos hinos,
com muitos hinos apresentando altos scores em emoções positivas e baixos em negativas -- quase uma linha horizontal, com
score negativas próximo de zero. No entanto, é interessante notar que alguns hinos compartilham scores positivos
e negativos moderados (exemplos: 258 -- "Se tu estás tão longe", 132 -- "Já não estou sozinho", e 
98 -- "Onde estaria eu?"), indicando uma complexidade emocional onde sentimentos mistos são expressos.
"""

# Valência emocional média: 0.701
f"""
A valência emocional média -- diferença entre positivas e negativas -- é de {hinos_analise['valencia_emocional'].mean():.3f}, 
sendo que quanto maior o valor, mais positivo são os hinos, e vice-versa. Pela análise do gráfico, podemos determinar que o 
hino mais positivo é o **{hinos_analise.loc[hinos_analise['valencia_emocional'].idxmax()]['nome']}**, enquanto o mais negativo é o 
**{hinos_analise.loc[hinos_analise['valencia_emocional'].idxmin()]['nome']}**.

"""

"""
#### Lista de hinos negativos

Como apenas 21 hinos foram classificados como negativos, segue a lista completa deles:
"""

hinos_negativos = hinos_analise[
    hinos_analise["categoria_dominante"] == "negativas"
].sort_values("valencia_emocional")
st.dataframe(
    hinos_negativos[
        ["nome", "valencia_emocional", "emocao_dominante", "texto_limpo"]
    ].rename_axis("Nº"),
    width="stretch",
    column_config={
        "nome": st.column_config.TextColumn(
            "Nome do Hino", help="Nome do hino negativo."
        ),
        "valencia_emocional": st.column_config.NumberColumn(
            "Valência Emocional",
            help="Diferença entre scores positivos e negativos.",
            width="small",
        ),
        "emocao_dominante": st.column_config.TextColumn(
            "Emoção Dominante", help="Emoção predominante no hino.", width="small"
        ),
        "texto_limpo": st.column_config.TextColumn(
            "Texto do Hino", help="Trecho do texto do hino.", width="large"
        ),
    },
)

"""
**Uma conclusão importante:** embora seja possível entender o motivo de alguns hinos serem classificados como negativos,
nem sempre é tão evidente, podendo ser até mesmo contraditório. Por vezes, pode ser uma interpretação incorreta por parte do modelo,
como no caso do hino 30 -- "Oh! Que precioso sange", cuja emoção dominante é tristeza ("sadness"), 
apesar do texto ser claramente positivo. Isso pode ocorrer por diversos fatores, como a presença de
linguagem figurada, metáforas, ou mesmo a ambiguidade inerente à linguagem humana. Portanto, é crucial interpretar esses resultados
com cautela, considerando o contexto mais amplo dos hinos e reconhecendo as limitações e possíveis erros 
dos modelos de análise de sentimentos.
"""

# casos extremos
"""
### Casos extremos 

Aqui, destacamos hinos que se sobressaem em diferentes aspectos emocionais, como os mais atípicos, típicos, 
negativos e balanceados. 
"""
emocoes_principais = (
    hinos_analise["emocao_dominante_sem_neutral"].value_counts().head(8).index
)

# Compact overview dos top hinos por emoção principal (substitui a lista longa)
rows = []
scores_matrix = []
hover_matrix = []
ranks = ["1", "2", "3"]

for emocao in emocoes_principais:
    # calcular scores temporários sem criar colunas no DF
    scores = hinos_analise["emocoes"].apply(lambda x: x.get(emocao, 0.0) if x else 0.0)
    top_idx = scores.nlargest(3).index.tolist()
    # preencher a linha do resumo textual
    rank_cells = []
    row_scores = []
    for ri in range(3):
        if ri < len(top_idx):
            idx = top_idx[ri]
            hino = hinos_analise.loc[idx]
            s = float(scores.loc[idx])
            # célula compacta para tabela
            rank_cells.append(f"{idx} - {hino['nome']} ({s:.3f})")
            # dados para heatmap / hover
            row_scores.append(s)
        else:
            rank_cells.append("")
            row_scores.append(0.0)
    rows.append(
        {"emocao": emocao, "1": rank_cells[0], "2": rank_cells[1], "3": rank_cells[2]}
    )
    scores_matrix.append(row_scores)

# Mostrar tabela compacta (emoção x top3)
df_top3 = pd.DataFrame(rows).set_index("emocao").rename_axis("Emoção")
"""
**Ranking de Hinos por Emoção Principal:**
A tabela abaixo apresenta os três hinos com os maiores scores para cada uma das 8 principais emoções identificadas.

"""
st.dataframe(
    df_top3,
    width="stretch",
    column_config={
        "1": st.column_config.TextColumn(
            "1º Lugar",
            help="Hino com maior score na emoção.",
            width="medium",
        ),
        "2": st.column_config.TextColumn(
            "2º Lugar",
            help="Hino com segundo maior score na emoção.",
            width="medium",
        ),
        "3": st.column_config.TextColumn(
            "3º Lugar",
            help="Hino com terceiro maior score na emoção.",
            width="medium",
        ),
    },
)

"""
É notável como pelo menos um hino de cada ranking concorda com a emoção dominante previamente identificada,
o que colabora com a consistência dos dados analisados. "Senhor, te amo, te amo" na categoria de "amor"; o otimismo de quem "Espera no Senhor, anima-te!"; 
a admiração expressa em "Lindo! Lindo! Lindo!"; a alegria de quem canta "Às vezes, alguém me 
pergunta"; a tristeza antes de ver que "Uma luz brilhou em meu caminho". Esses exemplos ilustram bem como as emoções 
são capturadas e refletidas nos hinos.
"""


# Calcular distância do perfil emocional médio
# Criar vetor de emoções médias
emocoes_todas = set()
for emocoes in hinos_analise["emocoes"]:
    if emocoes:
        emocoes_todas.update(emocoes.keys())

vetor_medio = {}
for emocao in emocoes_todas:
    scores = [e.get(emocao, 0.0) for e in hinos_analise["emocoes"] if e]
    vetor_medio[emocao] = np.mean(scores) if scores else 0.0


# Calcular distância euclidiana de cada hino para a média
def calcular_distancia_media(emocoes):
    if not emocoes:
        return 0.0
    vetor_hino = [emocoes.get(emocao, 0.0) for emocao in sorted(vetor_medio.keys())]
    vetor_medio_sorted = [vetor_medio[emocao] for emocao in sorted(vetor_medio.keys())]
    return euclidean(vetor_hino, vetor_medio_sorted)


hinos_analise["distancia_perfil_medio"] = hinos_analise["emocoes"].apply(
    calcular_distancia_media
)

col1, col2 = st.columns(2)

with col1:
    st.write("**Hinos MAIS ATÍPICOS (perfil emocional único)**")
    mais_atipicos = hinos_analise.nlargest(5, "distancia_perfil_medio")

    if mais_atipicos.empty:
        st.write("Nenhum hino atípico encontrado.")
    else:
        rows = []
        for i, (idx, hino) in enumerate(mais_atipicos.iterrows(), 1):
            rows.append(
                {
                    "Rank": i,
                    "Nome": f"{idx} - {hino['nome']}",
                    "Distância do perfil médio": round(
                        hino["distancia_perfil_medio"], 3
                    ),
                    "Emoção dominante": hino["emocao_dominante_sem_neutral"],
                }
            )
        df_atipicos = pd.DataFrame(rows).set_index("Rank")
        st.dataframe(
            df_atipicos,
            column_config={
                "Nome": st.column_config.TextColumn(
                    "Nome do Hino",
                    help="Identificação do hino pelo seu índice e nome.",
                    width="small",
                ),
                "Distância do perfil médio": st.column_config.NumberColumn(
                    "Distância do Perfil Médio",
                    help="Quão distante o perfil emocional do hino está do perfil médio da coletânea.",
                    width="small",
                ),
                "Emoção dominante": st.column_config.TextColumn(
                    "Emoção Dominante",
                    help="A emoção mais forte no hino.",
                    width="small",
                ),
            },
        )


with col2:
    st.write("**Hinos MAIS TÍPICOS (perfil emocional comum)**")
    mais_tipicos = hinos_analise.nsmallest(5, "distancia_perfil_medio")

    if mais_tipicos.empty:
        st.write("Nenhum hino típico encontrado.")
    else:
        rows = []
        for i, (idx, hino) in enumerate(mais_tipicos.iterrows(), 1):
            rows.append(
                {
                    "Rank": i,
                    "Nome": f"{idx} - {hino['nome']}",
                    "Distância do perfil médio": round(
                        hino["distancia_perfil_medio"], 3
                    ),
                    "Emoção dominante": hino["emocao_dominante_sem_neutral"],
                }
            )
        df_tipicos = pd.DataFrame(rows).set_index("Rank")
        st.dataframe(
            df_tipicos,
            column_config={
                "Nome": st.column_config.TextColumn(
                    "Nome do Hino",
                    help="Identificação do hino pelo seu índice e nome.",
                    width="small",
                ),
                "Distância do perfil médio": st.column_config.NumberColumn(
                    "Distância do Perfil Médio",
                    help="Quão distante o perfil emocional do hino está do perfil médio da coletânea.",
                    width="small",
                ),
                "Emoção dominante": st.column_config.TextColumn(
                    "Emoção Dominante",
                    help="A emoção mais forte no hino.",
                    width="small",
                ),
            },
        )

"""
De forma geral, hinos mais atípicos estão relacionados à emoção de "gratidão", enquanto hinos mais típicos
tendem a expressar "amor", que é a emoção mais comum na coletânea.
"""

col1, col2 = st.columns(2)

with col1:
    st.write("**Hinos MAIS NEGATIVOS**")
    hinos_negativos = hinos_analise.nlargest(5, "score_negativas")

    if hinos_negativos.empty:
        st.write("Nenhum hino típico encontrado.")
    else:
        rows = []
        for i, (idx, hino) in enumerate(hinos_negativos.iterrows(), 1):
            rows.append(
                {
                    "Rank": i,
                    "Nome": f"{idx} - {hino['nome']}",
                    "Score negativas": round(hino["score_negativas"], 3),
                    "Emoção dominante": hino["emocao_dominante_sem_neutral"],
                }
            )
        df_tipicos = pd.DataFrame(rows).set_index("Rank")
        st.dataframe(
            df_tipicos,
            column_config={
                "Nome": st.column_config.TextColumn(
                    "Nome do Hino",
                    help="Identificação do hino pelo seu índice e nome.",
                    width="small",
                ),
                "Score negativas": st.column_config.NumberColumn(
                    "Score negativas",
                    help="Quão distante o perfil emocional do hino está do perfil médio da coletânea.",
                    width="small",
                ),
                "Emoção dominante": st.column_config.TextColumn(
                    "Emoção Dominante",
                    help="A emoção mais forte no hino.",
                    width="small",
                ),
            },
        )

with col2:
    st.write("**Hinos com PERFIL MAIS BALANCEADO (múltiplas emoções fortes)**")
    # Esses são os com maior diversidade mas baixa concentração
    hinos_balanceados = hinos_analise.nsmallest(5, "concentracao_emocional")

    if hinos_balanceados.empty:
        st.write("Nenhum hino típico encontrado.")
    else:
        rows = []
        for i, (idx, hino) in enumerate(hinos_balanceados.iterrows(), 1):
            rows.append(
                {
                    "Rank": i,
                    "Nome": f"{idx} - {hino['nome']}",
                    "Concentração": round(hino["concentracao_emocional"], 3),
                    "Diversidade": round(hino["diversidade_emocional"], 3),
                }
            )

        df_tipicos = pd.DataFrame(rows).set_index("Rank")
        st.dataframe(
            df_tipicos,
            column_config={
                "Nome": st.column_config.TextColumn(
                    "Nome do Hino",
                    help="Identificação do hino pelo seu índice e nome.",
                    width="small",
                ),
                "Concentração": st.column_config.NumberColumn(
                    "Concentração",
                    help="Quão concentrado está o perfil emocional do hino.",
                    width="small",
                ),
                "Diversidade": st.column_config.NumberColumn(
                    "Diversidade",
                    help="A diversidade emocional do hino.",
                    width="small",
                ),
            },
        )


"""
Embora seja de conhecimento geral que os hinos tendem a ser positivos, com temas de consolo e encorajamento, vemos que
emoções negativas se fazem presentes também: o clamor contínuo em "Em tuas mãos, Senhor"; a tristeza em "Se anda triste
o teu viver" e "Quando andava no mundo escravizado".

Por fim, hinos com perfis emocionais balanceados, como "Eu não saberia caminhas" e "Lá onde eu não posso ver",
demonstram uma rica tapeçaria de emoções, refletindo a complexidade da experiência humana em sua relação com o divino.
"""

# resumo emocional
"""
## Resumo Emocional da Coletânea

Em suma, a coletânea pode ser emocionalmente caracterizada pelos seguintes aspectos:
"""

top_emocoes_geral = hinos_analise["emocao_dominante_sem_neutral"].value_counts().head(5)
cat_dist = hinos_analise["categoria_dominante"].value_counts()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total de hinos", len(hinos_analise))
    st.metric(
        "Valência média",
        f"{hinos_analise['valencia_emocional'].mean():.3f}",
        delta=(
            "POSITIVA" if hinos_analise["valencia_emocional"].mean() > 0 else "NEGATIVA"
        ),
    )
with col2:
    st.metric(
        "Intensidade média", f"{hinos_analise['intensidade_emocional'].mean():.3f}"
    )
    st.metric(
        "Diversidade média", f"{hinos_analise['diversidade_emocional'].mean():.3f}"
    )
with col3:
    st.metric(
        "Categoria mais comum",
        f"{cat_dist.index[0].upper()}",
        delta=f"{cat_dist.iloc[0]} hinos",
    )
    st.metric(
        "Emoção mais comum",
        f"{top_emocoes_geral.index[0].upper()}",
        delta=f"{top_emocoes_geral.iloc[0]} hinos",
    )

"""
A coletânea de hinos é predominantemente positiva, com uma valência média de 0.701, indicando um forte viés otimista. 
A emoção mais comum é "amor", refletindo temas de afeto e compaixão. A categoria emocional mais frequente é "positiva",
sugerindo que a maioria dos hinos visa inspirar esperança e alegria. A intensidade e diversidade emocionais médias indicam 
que os hinos são emocionalmente ricos, expressando uma ampla gama de sentimentos de maneira significativa.
"""

"""
### Hinos mais semelhantes emocionalmente

A seguir, selecione um hino para ver os mais semelhantes com base no perfil emocional.
"""

hinos_opcoes = [f"{num} - {row['nome']}" for num, row in hinos_analise.iterrows()]
hino_selecionado = st.selectbox(
    "Pesquisar hino (número ou nome)",
    options=hinos_opcoes,
    placeholder="Digite para buscar...",
    index=None,
    help="Digite o número ou parte do nome do hino para pesquisar",
)
if hino_selecionado:
    hymn_num = int(hino_selecionado.split(" - ")[0])
    hymn_name = hinos_analise.loc[hymn_num, "nome"]

    st.metric(label="🎵 Hino", value=f"{hymn_num} — {hymn_name}")

    f"""
    **Emocoes principais:** {', '.join([f'{k}({v:.2f})' for k, v in sorted(hinos_analise.loc[hymn_num, 'emocoes'].items(), key=lambda x: x[1], reverse=True)[:3]])}
    """

    similarities = list(enumerate(similarity_emocoes.loc[hymn_num]))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    rows = []
    for idx, score in similarities[1:11]:
        rows.append(
            {
                "Hino": int(idx),
                "Nome": hinos_analise["nome"].iloc[idx],
                "Similaridade": float(score),
            }
        )
    df_sim = pd.DataFrame(rows).set_index("Hino")
    st.dataframe(df_sim.style.format({"Similaridade": "{:.3f}"}))
else:
    st.info("Selecione um hino acima para ver os mais semelhantes emocionalmente.")