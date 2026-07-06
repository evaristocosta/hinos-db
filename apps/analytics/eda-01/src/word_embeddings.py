import streamlit as st
import pandas as pd
from src.loader import hinos_processados, similarity_matrices
import plotly.express as px
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

#    Word embeddings (eda1_part4):
st.title("📝 Embeddings de Palavras")

"""
Nesta seção, exploramos os embeddings de palavras gerados a partir dos textos dos hinos.
Utilizamos técnicas de processamento de linguagem natural para transformar os textos em representações 
vetoriais densas, que capturam o significado semântico das palavras. Esses embeddings permitem analisar 
similaridades entre hinos, realizar clustering e identificar tópicos comuns.
"""

hinos_analise: pd.DataFrame = hinos_processados()
hinos_analise["word_cluster"] = hinos_analise["word_cluster"].astype("category")
hinos_analise["NMF_topic"] = hinos_analise["NMF_topic"].astype("category")
# hinos_analise = hinos_analise.rename_axis("Nº")
_, similarity_word, _, _ = similarity_matrices()

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

# - Matrizes de similaridade com heatmap
# estratégia de peso: TF-IDF
"""
## Matriz de Similaridade entre Hinos

Aqui, visualizamos a matriz de similaridade entre os hinos com base nos embeddings de palavras, calculada a partir da 
estratégia de peso TF-IDF. Cada célula na matriz representa o grau de similaridade entre dois hinos, onde valores 
mais altos indicam maior similaridade semântica.

"""

st.warning(
    "Aplicar filtros pode causar problemas na visualização da matriz de similaridade.",
    icon="⚠️",
)

# filtrar a matriz de similaridade para mostrar apenas os hinos presentes em hinos_analise
selected_idx = list(hinos_analise.index)

# se similarity_word for DataFrame, usamos .loc; caso seja ndarray, convertemos para DataFrame com índices posicionais
if hasattr(similarity_word, "loc"):
    sim_filtered = similarity_word.loc[selected_idx, selected_idx]
else:
    sim_df = pd.DataFrame(
        similarity_word,
        index=range(similarity_word.shape[0]),
        columns=range(similarity_word.shape[0]),
    )
    sim_filtered = sim_df.loc[selected_idx, selected_idx]

sim_filtered = sim_filtered.sort_index(axis=0).sort_index(axis=1)
fig = px.imshow(
    sim_filtered,
    x=sim_filtered.columns.astype(str),
    y=sim_filtered.index.astype(str),
    labels=dict(x="Hinos", y="Hinos", color="Similaridade"),
    width=600,
    height=600,
    color_continuous_scale="Cividis",
)
st.plotly_chart(fig)

"""
Diferentemente da matriz de similaridade baseada em TF-IDF, que se concentra na frequência e importância das palavras 
nos documentos, a matriz de similaridade baseada em embeddings de palavras captura relações semânticas mais profundas 
entre os hinos. Isso significa que hinos com significados semelhantes, mesmo que usem palavras diferentes, podem ser 
identificados como similares, o que explica porque mais hinos aparecem como similares nesta matriz.

Dois hinos que chamam a atenção são 106 -- "Pela fé somos salvos" e 179 -- "Pela fé eu posso contemplar Jesus": ambos tem 
baixa similaridade com a maioria dos outros hinos, mas alta similaridade entre si. Isso sugere que, apesar de usarem 
palavras diferentes, eles compartilham um significado semântico semelhante, relacionado ao tema da fé e salvação.
"""

# - Hinos mais semelhantes
"""
### Hinos mais semelhantes

A seguir, selecione um hino para ver os mais semelhantes com base nos embeddings de palavras.
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

    similarities_tfidf = list(enumerate(similarity_word.loc[hymn_num]))
    similarities_tfidf = sorted(similarities_tfidf, key=lambda x: x[1], reverse=True)

    rows = []
    for idx, score in similarities_tfidf[1:11]:
        rows.append(
            {
                "Hino": int(idx),
                "Nome": hinos_analise["nome"].loc[idx],
                "Similaridade": float(score),
            }
        )
    df_sim = pd.DataFrame(rows).set_index("Hino")
    st.dataframe(df_sim.style.format({"Similaridade": "{:.3f}"}))
else:
    st.caption("Selecione um hino para ver os mais semelhantes.")

# - Clustering
# Diminuição de dimensionalidade: UMAP
# Clustering: K-Means
# Definição de clusters: silhueta - 4º melhor valor, 10 clusters
f"""
## Clustering de Hinos

Utilizando os embeddings de palavras, aplicamos técnicas de redução de dimensionalidade (UMAP) e clustering (K-Means) para 
agrupar os hinos com base em suas similaridades semânticas. A visualização abaixo mostra os hinos em um espaço bidimensional,
onde cores diferentes representam clusters (ou grupos) distintos. 

Cada ponto representa um hino, e a proximidade entre os pontos indica similaridade semântica. Clusters próximos
sugerem temas ou estilos comuns entre os hinos. 

A definição do número de clusters foi baseada na análise de silhueta, resultando em {hinos_analise['word_cluster'].nunique()} 
clusters que capturam bem as  variações nos temas dos hinos -- embora o melhor valor de silhueta tenha sido encontrado para 3 clusters, decidi 
manter 10 clusters para uma representação mais granular e próxima da quantidade de categorias originais que existem na coletânea.
"""

fig = px.scatter(
    hinos_analise,
    x="word_umap1",
    y="word_umap2",
    color="word_cluster",
    hover_data=["nome"],
    # title="Clustering de Hinos com Embeddings de Palavras",
    labels={"word_umap1": "", "word_umap2": "", "word_cluster": "Cluster"},
    width=600,
    height=600,
)
st.plotly_chart(fig)

"""
O agrupamento resultante permite dividir claramente os hinos em diferentes categorias. No geral, todos os hinos partilham
de um espaço comum. No entanto, alguns clusters se destacam por sua separação mais clara, indicando temas ou estilos únicos.
Por exemplo, os clusters 2 (temática de "rei") e 7 (temática de "vem") tem parte no espaço comum, mas também possuem áreas distintas, sugerindo que embora
compartilhem algumas características com outros hinos, eles também possuem elementos únicos que os diferenciam. 
"""

# - Termos mais frequentes por cluster
"""
### Termos mais frequentes por cluster

A seguir, apresentamos os primeiros 8 termos mais frequentes em cada cluster de hinos, conforme identificado pelo 
algoritmo de clustering. Esses termos fornecem insights sobre os temas predominantes em cada grupo de hinos. 
"""


rows = []
for c in sorted(hinos_analise["word_cluster"].unique()):
    cluster_tokens = hinos_analise.loc[
        hinos_analise["word_cluster"] == c, "tokens_no_stops"
    ].sum()
    top_terms = [t for t, _ in Counter(cluster_tokens).most_common(8)]
    cluster_series = hinos_analise.loc[hinos_analise["word_cluster"] == c, "nome"]
    sampled = cluster_series.sample(n=min(3, len(cluster_series)), random_state=42)
    top_hinos = [f"{int(idx)} - {name}" for idx, name in sampled.items()]
    rows.append(
        {
            "Cluster": str(c),
            "Termos": ", ".join(top_terms),
            "Hinos de exemplo": " | ".join(top_hinos),
        }
    )

df_clusters = pd.DataFrame(rows).set_index("Cluster")
st.dataframe(df_clusters)


"""
É notável que algumas palavras-chave, como "Senhor", "Jesus" e "Deus", aparecem frequentemente em múltiplos clusters,
indicando sua importância central nos temas dos hinos. Como vimos em análises anteriores, essas são as palavras
mais comuns em todo o corpus de hinos.
"""


hinos_cluster7 = hinos_analise[hinos_analise["word_cluster"] == 7][
    ["nome", "categoria_abr"]
].rename_axis("Nº")
f"""
#### Cluster 7 em perspectiva

Para exemplificar e aprofundar a análise, vamos observar o cluster 7. 
O cluster 7 é composto por um total de {hinos_cluster7.shape[0]} hinos. Vimos que este cluster está relacionado a temas que
falam da volta de Jesus. A seguir, é apresentada a contagem de hinos por categoria dentro deste cluster, permitindo identificar quais 
categorias da coletânea estão mais representadas.
"""
df_cluster7_cat = (
    hinos_cluster7["categoria_abr"]
    .value_counts()
    .sort_values(ascending=True)
    .reset_index()
)
df_cluster7_cat.columns = ["Categoria", "Quantidade"]
fig = px.bar(
    df_cluster7_cat,
    x="Quantidade",
    y="Categoria",
    orientation="h",
    labels={"Quantidade": "Quantidade de Hinos", "Categoria": "Categoria"},
    color_discrete_sequence=["#6181a8"],
)
st.plotly_chart(fig)

f"""
As duas categorias mais presentes são de "VOLTA DE JESUS E ETERNIDADE" e "SANTIFICAÇÃO E DERRAMAMENTO DO ESPÍRITO SANTO", com 17 e 
12 hinos, respectivamente. Confirmando os dados analisados anteriormente, o cluster 7 é fortemente associado a temas de volta de Jesus,
com {17/hinos_cluster7.shape[0]*100:.1f}% dos hinos do cluster pertencendo a essa categoria. Não obstante, podemos ainda inferir que existem
hinos que falam sobre a volta de Jesus mas estão classificados em outras categorias da coletânea.
"""


""" 
### Relação entre Clusters e Categorias da Coletânea


Analisamos a distribuição dos clusters de hinos em relação às categorias originais da coletânea. Isso nos ajuda a entender como os 
agrupamentos baseados em embeddings de palavras correspondem às categorias pré-definidas. A seguir, apresentamos uma visualização 
que mostra a proporção de hinos de cada categoria dentro de cada cluster.
"""

# tabela de contingência: categorias x clusters
ct = pd.crosstab(
    hinos_analise["categoria_abr"], hinos_analise["word_cluster"]
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
        "y": "Cluster",
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
Podemos observar que alguns clusters têm alguma associação com categorias específicas da coletânea. Por exemplo, o cluster 8 tem uma forte presença 
de hinos da categoria "CLAMOR", enquanto o cluster 9 é dominado por hinos da categoria "SALMOS DE LOUVOR". No entanto, muitos clusters apresentam 
uma distribuição mais diversificada de categorias, indicando que os agrupamentos baseados em embeddings de palavras não correspondem diretamente às 
categorias pré-definidas. Isso sugere que os embeddings capturam nuances semânticas que transcendem as categorias tradicionais, refletindo a 
complexidade dos temas abordados nos hinos.
"""


# - Tópicos comuns
"""
## Tópicos comuns entre os hinos

Utilizando a técnica de Non-negative Matrix Factorization (NMF) aplicada à representação TF-IDF dos textos dos hinos,
identificamos tópicos comuns presentes nos hinos. Usamos o número de clusters previamente definido como o número de 
tópicos para garantir consistência na análise.
A seguir, apresentamos os principais tópicos e suas palavras-chave associadas.
"""


n_topics = hinos_analise["word_cluster"].nunique()

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


def display_topics(model, feature_names, n_top_words=10):
    rows = []
    for idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[: -n_top_words - 1 : -1]]
        rows.append({"Tópico": f"{idx+1}", "Palavras-chave": ", ".join(top_words)})
    df_topics = pd.DataFrame(rows).set_index("Tópico")
    return df_topics


feature_names = vectorizer.get_feature_names_out()
df_topics = display_topics(nmf, feature_names)
st.dataframe(df_topics)

"""
Diferentemente dos termos mais frequentes por cluster, os tópicos identificados pelo NMF consideram também bigramas e trigramas,
o que pode revelar temas mais específicos e contextuais presentes nos hinos. Por exemplo, o tópico 3 está diretamente
relacionado ao tema de "Volta de Jesus", enquanto que os tópicos 8 e 10 são de "Glória" e "Aleluia", respectivamente.
Ainda, o tópico 5 contém termos relacionados ao clamor pelo sangue de Jesus, e o tópico 9 indica hinos de serviço e adoração.
"""

# - Distribuição de tópicos
"""
### Distribuição de Tópicos nos Hinos

Podemos usar os tópicos identificados para analisar a distribuição dos hinos no espaço UMAP, colorindo-os de acordo com o tópico
dominante atribuído pelo NMF.
"""

fig = px.scatter(
    hinos_analise,
    x="word_umap1",
    y="word_umap2",
    labels={"word_umap1": "", "word_umap2": "", "NMF_topic": "Tópico NMF"},
    color="NMF_topic",
    hover_data=["nome"],
    width=600,
    height=600,
)
st.plotly_chart(fig)

"""
Diferente dos claros agrupamentos observados com o clustering baseado em K-Means, a distribuição dos tópicos NMF no espaço UMAP
é mais difusa. Isso sugere que os tópicos identificados pelo NMF capturam nuances semânticas que não se traduzem 
diretamente em clusters distintos, indicando uma sobreposição maior entre os temas dos hinos.

É possível observar algumas relações, no entanto. Por exemplo, os hinos do tópico 2 aparecem na mesma região do espaço UMAP
associada ao cluster 7, e o tópico 6 está presente em uma área próxima ao cluster 4.
"""

hinos_topico2 = hinos_analise[hinos_analise["NMF_topic"] == 2][
    ["nome", "categoria_abr"]
].rename_axis("Nº")

f"""
#### Hinos do Tópico 2

O tópico 2 é composto por um total de {hinos_topico2.shape[0]} hinos, {"mais" if hinos_topico2.shape[0] > hinos_cluster7.shape[0] else "menos"} 
do que o cluster 7 (que tem {hinos_cluster7.shape[0]} hinos). A seguir, apresentamos as categorias a que pertencem os hinos do tópico 2.
"""

df_topico2_cat = (
    hinos_topico2["categoria_abr"]
    .value_counts()
    .sort_values(ascending=True)
    .reset_index()
)
df_topico2_cat.columns = ["Categoria", "Quantidade"]
fig = px.bar(
    df_topico2_cat,
    x="Quantidade",
    y="Categoria",
    orientation="h",
    labels={"Quantidade": "Quantidade de Hinos", "Categoria": "Categoria"},
    color_discrete_sequence=["#6181a8"],
)
st.plotly_chart(fig)


"""
Observa-se que os hinos do tópico 2 abrangem diversas categorias, sendo que as mais marcantes são sobre volta de Jesus
e santificação, assim como o cluster 7. No entanto, o tópico 2 inclui hinos em mais categorias do que o cluster 7, refletindo a natureza 
mais difusa dos tópicos NMF em comparação com os clusters.
"""
