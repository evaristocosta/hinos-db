import streamlit as st
import pandas as pd
from src.loader import hinos_processados, similarity_matrices
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import numpy as np

#    Sequence embeddings (eda1_part5):
st.title("🗒️ Embeddings de frases")
"""
Nesta seção, exploramos os embeddings de frases gerados a partir dos hinos. Os embeddings são representações 
vetoriais que capturam o significado semântico de frases inteiras ao invés de palavras isoladas, permitindo 
comparações e análises mais profundas.
"""

hinos_analise: pd.DataFrame = hinos_processados()
hinos_analise["sent_cluster"] = hinos_analise["sent_cluster"].astype("category")
hinos_analise["BERT_topic"] = hinos_analise["BERT_topic"].astype("category")
_, _, similarity_sentence, _ = similarity_matrices()

st.sidebar.header("Filtros")
categorias_unicas = hinos_analise["categoria_abr"].unique()
categorias_selecionadas = st.sidebar.multiselect(
    "Selecione as categorias",
    options=categorias_unicas,
    placeholder="Todas as categorias",
)
if categorias_selecionadas:
    hinos_analise = hinos_analise[
        hinos_analise["categoria_abr"].isin(categorias_selecionadas)
    ]


# modelo = SentenceTransformer("rufimelo/Legal-BERTimbau-sts-base")  # português brasileiro
# similaridade = cosine_similarity

"""
## Matriz de Similaridade entre Hinos

Como na análise de embeddings de palavras, aqui apresentamos a matriz de similaridade entre os hinos,
mas agora utilizando os embeddings de frases. 

Para gerar os embeddings de frases, utilizamos o modelo "[rufimelo/Legal-BERTimbau-sts-base](https://huggingface.co/rufimelo/Legal-BERTimbau-sts-base)", 
que é baseado na arquitetura BERT e foi ajustado para tarefas de similaridade semântica em português brasileiro.
A similaridade por sua vez, é calculada usando a similaridade do cosseno.

Um ponto importante é que os embeddings de frases são gerados a partir do texto completo de cada hino,
e não apenas de palavras individuais -- processo de tokenização e remoção de stopwords não são aplicados aqui.

"""

st.warning(
    "Aplicar filtros pode causar problemas na visualização da matriz de similaridade.",
    icon="⚠️",
)

# restringe a matriz de similaridade aos hinos atualmente no dataframe (caso haja filtro)
idx = hinos_analise.index.tolist()
sim_sub = similarity_sentence.loc[idx, idx]

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
Comparando com a matriz de similaridade baseada em embeddings de palavras, podemos observar que a matriz
de embeddings de frases tende a apresentar valores de similaridade mais altos entre os hinos. Isso ocorre porque os 
embeddings de frases capturam o contexto completo das frases, levando em consideração a estrutura e o significado 
global, enquanto os embeddings de palavras focam em palavras individuais.
Essa característica dos embeddings de frases permite identificar similaridades semânticas mais profundas entre os 
hinos, mesmo quando eles utilizam palavras diferentes para expressar ideias semelhantes.

O que chama atenção, pela análise visual da matriz, são algumas linhas e colunas mais "azuis", indicando hinos que
não compartilham muita similaridade com os demais, como o hino 396 -- "Abba Pai", ou 13 -- "Vamos lavar as vestes". 
Esses hinos podem ser considerados mais únicos em termos de conteúdo e estilo, destacando-se na coletânea.
A região dos corinhos (maiores que 731) também se destaca, mostrando hinos de menor similaridade com os demais, e
mesmo entre eles. De fato, são hinos característicos, com estruturas e temas próprios, o que justifica sua menor 
similaridade. O mesmo acontece com alguns hinos de clamor, e de invocação. No entanto, a faixa que mais chama atenção
é o intervalo entre os hinos 396 e 403, que apresentam uma similaridade muito baixa com o restante dos hinos. Um
fator que pode ser determinante, é que esses hinos são mais curtos, com menos versos, o que pode influenciar
na geração dos embeddings e na similaridade calculada.
"""


"""
### Relação de tamanho do hino e similaridade

Aqui, investigamos se existe alguma correlação entre o tamanho dos hinos (medido pelo número de tokens)
e a similaridade média com os demais hinos, utilizando os embeddings de frases.
"""


# restringe a matriz de similaridade aos hinos atualmente no dataframe (caso haja filtro)
idx = hinos_analise.index.tolist()
sim_sub = similarity_sentence.loc[idx, idx]

# média de similaridade com os demais (exclui a diagonal / self-similarity)
n = sim_sub.shape[0]
if n > 1:
    mean_sim = (sim_sub.sum(axis=1) - np.diag(sim_sub).astype(float)) / (n - 1)
else:
    mean_sim = pd.Series(0.0, index=sim_sub.index)


# conta número de tokens (compatível com listas ou strings)
def _count_tokens(x):
    try:
        return len(x)
    except Exception:
        if pd.isna(x):
            return 0
        return len(str(x).split())


size_series = hinos_analise["tokens_no_stops"].apply(_count_tokens)

plot_df = pd.DataFrame(
    {
        "hino": hinos_analise.index,
        "nome": hinos_analise["nome"],
        "tamanho": size_series,
        "similaridade_media": mean_sim.loc[hinos_analise.index].astype(float),
    }
).reset_index(drop=True)

# calcula correlação e ajuste linear
mask = np.isfinite(plot_df["tamanho"]) & np.isfinite(plot_df["similaridade_media"])
corr = np.corrcoef(
    plot_df.loc[mask, "tamanho"], plot_df.loc[mask, "similaridade_media"]
)[0, 1]

# scatter + linha de regressão
fig = px.scatter(
    plot_df,
    x="tamanho",
    y="similaridade_media",
    hover_data=["hino", "nome"],
    labels={
        "tamanho": "Número de tokens (tamanho do hino)",
        "similaridade_media": "Similaridade média",
    },
    title="Relação entre tamanho do hino e similaridade média",
    width=700,
    height=450,
    color_discrete_sequence=["#6181a8"],
)
st.plotly_chart(fig)


f"""
Fica claro que a afirmação anterior sobre hinos mais curtos terem menor similaridade se confirma aqui. Embora existam hinos
com baixo número de tokens que apresentam similaridade média alta, a tendência geral indica que hinos mais curtos tendem a ter
menor similaridade média com os demais hinos. Isso pode ser atribuído ao fato de que hinos mais curtos possuem menos conteúdo
semântico para capturar, o que pode resultar em embeddings menos informativos e, consequentemente, em menor similaridade 
com outros hinos.

A **Correlação (Pearson)** entre tamanho e similaridade média é igual a {corr:.3f}.
Isso indica uma correlação positiva moderada, sugerindo que, em geral, hinos maiores tendem a ter similaridade média mais alta
com os demais hinos, embora existam exceções individuais.
"""


# mostra amostra dos valores
# st.dataframe(plot_df.sort_values("tamanho").head(10).set_index("hino"))


"""
### Hinos mais semelhantes

Usando os dados de similaridade, a seguir você pode selecionar um hino para ver os mais semelhantes com base 
nos embeddings de sentenças.
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

    similarities = list(enumerate(similarity_sentence.iloc[hymn_num]))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    results = [
        (idx, hinos_analise["nome"].iloc[idx], score)
        for idx, score in similarities[1:11]
    ]
    df_sim = (
        pd.DataFrame(results, columns=["hino", "nome", "similaridade"])
        .set_index("hino")
        .rename_axis("Nº")
    )
    df_sim["similaridade"] = df_sim["similaridade"].round(3)
    st.dataframe(
        df_sim,
        column_config={"nome": "Nome", "similaridade": "Similaridade"},
    )
else:
    st.info("Selecione um hino para ver os mais semelhantes.")

"""
## Clustering de Hinos com Embeddings de Sentenças

Assim como na análise de embeddings de palavras, aplicamos técnicas de redução de dimensionalidade (UMAP)
e clustering (K-Means) para visualizar e agrupar os hinos com base em seus embeddings de frases. Levando em conta
resultados da análise de silhueta, optei por 9 clusters para os embeddings de frases.
"""

fig = px.scatter(
    hinos_analise,
    x="sent_umap1",
    y="sent_umap2",
    color="sent_cluster",
    hover_data=["nome"],
    # title="Clustering de Hinos com Embeddings de Sentenças",
    labels={"sent_umap1": "", "sent_umap2": "", "sent_cluster": "Cluster"},
    width=600,
    height=600,
)
st.plotly_chart(fig)

"""
Na análise anterior, podíamos observar alguns hinos bem isolados em termos de similaridade. Aqui, vemos um agrupamento
mais coeso, com menos pontos isolados. Isso sugere que os embeddings de frases capturam melhor as semelhanças semânticas 
entre os hinos, permitindo uma formação de clusters mais definida.
"""


"""
### Termos mais frequentes por cluster

Aqui, apresentamos os termos mais frequentes em cada cluster de hinos baseado nos embeddings de frases, bem como hinos
representativos de cada cluster. 

"""


rows = []
for c in sorted(hinos_analise["sent_cluster"].unique()):
    cluster_tokens = hinos_analise.loc[
        hinos_analise["sent_cluster"] == c, "tokens_no_stops"
    ].sum()
    top_terms = [t for t, _ in Counter(cluster_tokens).most_common(8)]
    cluster_series = hinos_analise.loc[hinos_analise["sent_cluster"] == c, "nome"]
    sampled = cluster_series.sample(n=min(3, cluster_series.shape[0]))
    top_hymns = [f"{idx} - {name}" for idx, name in sampled.items()]
    rows.append(
        {
            "Cluster": c,
            "Termos": ", ".join(top_terms),
            "Hinos de exemplo": " | ".join(top_hymns),
        }
    )

df_terms = pd.DataFrame(rows).set_index("Cluster")
st.dataframe(df_terms)

"""
Embora embeddings de frases usem o texto completo dos hinos, incluindo stopwords, os termos mais frequentes em cada cluster 
ainda refletem temas centrais dos hinos agrupados. Vemos a presença de "Jesus", "Deus" e "Senhor" em todos os clusters,
sendo essas as palavras mais comuns na coletânea. Outros termos frequentes, como "amor", "glória", "aleluia" e "vida",
também aparecem, indicando temas recorrentes nos hinos. O cluster 1, por exemplo, é o único a destacar "sangue", sugerindo
hinos da categoria de "CLAMOR".
"""


"""
### Relação entre Clusters e Categorias da Coletânea

Como anteriormente, usando embeddings de palavras, analisamos a distribuição dos clusters de embeddings de sentenças de hinos 
em relação às categorias originais da coletânea. Assim, podemos entender como os agrupamentos baseados em embeddings de frases
correspondem às categorias pré-definidas. A seguir, apresentamos uma visualização que mostra a proporção de hinos de cada 
categoria dentro de cada cluster.
"""

# tabela de contingência: categorias x clusters
ct = pd.crosstab(
    hinos_analise["categoria_abr"], hinos_analise["sent_cluster"]
).sort_index()

# Heatmap (proporções por categoria) com anotações dentro dos quadrados
ct_counts = ct.copy()
ct_prop = ct_counts.div(
    ct_counts.sum(axis=1), axis=0
)  # normaliza por categoria (linha)
ct_prop_pct = ct_prop * 100  # em porcentagem

x = ct.index.tolist()  # categorias
y = [str(c) for c in ct.columns]  # clusters (string para rótulos)

fig_ct = px.imshow(
    ct_prop_pct.T.values,
    x=x,
    y=y,
    labels={
        "x": "Categoria da Coletânea",
        "y": "Cluster (sent_cluster)",
        "color": "Proporção (%)",
    },
    color_continuous_scale="Cividis",
    width=800,
    height=420,
)

# adicionar anotações com porcentagem e contagem
z = ct_prop_pct.T.values
counts = ct_counts.T.values
z_max = z.max() if z.size else 0
for i_y, y_label in enumerate(y):
    for i_x, x_label in enumerate(x):
        val_pct = z[i_y, i_x]
        cnt = int(counts[i_y, i_x])
        text = f"{val_pct:.1f}%\n({cnt})"
        # escolha de cor do texto para legibilidade
        text_color = "white" if val_pct > (z_max / 2 if z_max > 0 else 0.5) else "black"
        fig_ct.add_annotation(
            x=x_label,
            y=y_label,
            text=text,
            showarrow=False,
            font=dict(color=text_color, size=11),
            xanchor="center",
            yanchor="middle",
        )

fig_ct.update_layout(margin=dict(l=40, r=40, t=40, b=40))
st.plotly_chart(fig_ct)


"""
Podemos perceber que os clusters formados pelos embeddings de frases não apresentam uma correspondência direta com as categorias 
pré-definidas, ainda mais do que os clusters baseados em embeddings de palavras. Uma exceção é o cluster 3, que contém mais da metade
dos hinos na categoria "SALMOS DE LOUVOR". Interessantemente, o cluster 1 tem maior concentração de hinos da categoria
"MORTE, RESSURREIÇÃO E VIDA ETERNA", diferente da conclusão anterior de termos frequentes, onde o cluster 1 destacava 
"sangue", associado a "CLAMOR".
Portanto, embora os embeddings de frases capturem o significado semântico dos hinos, os 
agrupamentos resultantes não refletem necessariamente as categorias originais da coletânea. Isso sugere que os critérios utilizados 
para definir as categorias da coletânea podem ser diferentes dos aspectos semânticos capturados pelos embeddings de frases.

"""

# Obtenção de tópicos: BERTopic(embedding_model=model)
"""
## Tópicos comuns entre os hinos

Usando a técnica BERTopic, identificamos tópicos comuns entre os hinos com base nos embeddings de frases. Cada tópico é representado 
por um conjunto de palavras-chave que capturam o tema central dos hinos associados a esse tópico. Os tópicos não estão relacionados 
com os clusters anteriores, mas sim com temas semânticos extraídos dos textos dos hinos.

"""

topics = {
    0: ["amor", "me", "meu", "eu", "que", "em", "senhor", "mim", "quero", "teu"],
    1: ["glória", "de", "jesus", "que", "vem", "os", "com", "senhor", "santo", "rei"],
    2: ["eu", "que", "jesus", "cristo", "céu", "de", "me", "meu", "com", "dia"],
    3: ["que", "no", "ele", "de", "jesus", "deus", "na", "com", "do", "se"],
    4: [
        "senhor",
        "teu",
        "nos",
        "nosso",
        "nós",
        "nossa",
        "tua",
        "vidas",
        "louvor",
        "te",
    ],
    5: ["ti", "mim", "tu", "és", "minha", "meu", "de", "senhor", "em", "vem"],
    6: ["eu", "de", "meu", "hei", "ao", "do", "que", "ver", "me", "terra"],
    7: [
        "tais",
        "que",
        "dos",
        "sossegai",
        "um",
        "nos",
        "cristo",
        "senhor",
        "jesus",
        "deixa",
    ],
    8: ["sangue", "teu", "mim", "estendeu", "me", "para", "mão", "em", "sem", "senhor"],
    9: [
        "louvai",
        "senhor",
        "jerusalém",
        "aleluia",
        "do",
        "ao",
        "nome",
        "amém",
        "dos",
        "seja",
    ],
}

rows = [
    {"Tópico": f"{k}", "Palavras-chave": ", ".join(v)}
    for k, v in sorted(topics.items())
]
df_topics = pd.DataFrame(rows).set_index("Tópico")

st.table(df_topics)

"""
Aqui podemos ver uma maior presença de stopwords entre os termos mais frequentes de cada tópico, o que é esperado
já que os embeddings de frases consideram o texto completo dos hinos, incluindo essas palavras. No entanto, mesmo com a presença de stopwords, 
os tópicos ainda refletem temas centrais da coletânea. Um tópico que me chamou a atenção foi o 7, que inclui o termo "sossegai", um termo 
incomum na coletânea, provavelmente relacionado a um único hino: 310 - Mestre, o mar se revolta.
"""

# - Distribuição de tópicos
"""
### Distribuição de Tópicos nos Hinos

Utilizando os tópicos identificados pelo BERTopic, visualizamos a distribuição dos hinos em relação a esses tópicos. Vários pontos
estão marcados com valor igual a -1: isso indica que esses hinos não foram atribuídos a nenhum tópico específico pelo modelo,
sendo considerados "outliers" ou hinos que não se encaixam bem em nenhum dos tópicos identificados.

"""
st.info(
    "Na legenda do gráfico, é possível clicar no tópico -1 para ocultar esses pontos e melhorar a visualização."
)

fig = px.scatter(
    hinos_analise,
    x="sent_umap1",
    y="sent_umap2",
    color="BERT_topic",
    hover_data=["nome"],
    labels={"sent_umap1": "", "sent_umap2": "", "BERT_topic": "Tópico BERT"},
    width=600,
    height=600,
)
st.plotly_chart(fig)

"""
Interessantemente, podemos observar agrupamentos definidos para alguns tópicos, diferente do resultado da análise de tópicos para embeddings
de palavras. Inclusive, concordam com os agrupamentos vistos nos clusters de embeddings de frases. Por exemplo, o tópico 1, relacionado a "glória" e "santo",
está fortemente associado aos clusters 2 e 3, que também destacam esses termos. Da mesma forma, o tópico 2, centrado em "Jesus", "Cristo" e "céu", corresponde ao cluster 8,
que também enfatiza esses temas. Essa concordância sugere que os tópicos extraídos pelos embeddings de frases capturam aspectos semânticos semelhantes aos
identificados pelos clusters, reforçando a validade dos agrupamentos observados.

"""
