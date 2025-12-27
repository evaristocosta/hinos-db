import streamlit as st
import pandas as pd
from src.loader import hinos_processados, similarity_matrices
from topsis_hamedbaziyad import TOPSIS


#    TOPSIS (eda1_part6):

st.title("✅ Seleção de similares")
hinos_analise: pd.DataFrame = hinos_processados()
similarity_titles, similarity_word, similarity_sent, similarity_emocoes = (
    similarity_matrices()
)

"""
Nesta seção, utilizamos o método TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) para 
selecionar hinos similares com base em múltiplos critérios, incluindo categorias, clusters e similaridades, calculados
nas análises anteriores. O TOPSIS é uma técnica de tomada de decisão multi-critério que classifica as alternativas 
com base na distância de uma solução ideal positiva e negativa. 

"""


# - Funcionamento
# - Escolha dos pesos
"""## Defina os pesos para cada critério

Escolha uma configuração predefinida ou ajuste manualmente os pesos:

"""

# Presets de pesos
"""#### Presets de configuração"""

presets = {
    "Padrão": {
        "categoria_id": 100,
        "word_cluster": 90,
        "NMF_topic": 80,
        "sent_cluster": 70,
        "BERT_topic": 60,
        "sim_title": 50,
        "sim_word": 40,
        "sim_sent": 30,
        "sim_emocao": 20,
    },
    "Por Categoria": {
        "categoria_id": 100,
        "word_cluster": 20,
        "NMF_topic": 20,
        "sent_cluster": 20,
        "BERT_topic": 20,
        "sim_title": 10,
        "sim_word": 10,
        "sim_sent": 10,
        "sim_emocao": 10,
    },
    "Simil. Semântica": {
        "categoria_id": 10,
        "word_cluster": 30,
        "NMF_topic": 30,
        "sent_cluster": 100,
        "BERT_topic": 100,
        "sim_title": 90,
        "sim_word": 90,
        "sim_sent": 90,
        "sim_emocao": 20,
    },
    "Foco em Emoções": {
        "categoria_id": 20,
        "word_cluster": 20,
        "NMF_topic": 20,
        "sent_cluster": 30,
        "BERT_topic": 30,
        "sim_title": 30,
        "sim_word": 30,
        "sim_sent": 30,
        "sim_emocao": 100,
    },
    "Balanceado": {
        "categoria_id": 50,
        "word_cluster": 50,
        "NMF_topic": 50,
        "sent_cluster": 50,
        "BERT_topic": 50,
        "sim_title": 50,
        "sim_word": 50,
        "sim_sent": 50,
        "sim_emocao": 50,
    },
}

preset_cols = st.columns(len(presets))
for idx, (preset_name, preset_weights) in enumerate(presets.items()):
    if preset_cols[idx].button(preset_name, use_container_width=True):
        for key, value in preset_weights.items():
            st.session_state[key] = value
        st.rerun()

st.divider()

categories = [
    "categoria_id",
    "word_cluster",
    "NMF_topic",
    "sent_cluster",
    "BERT_topic",
    "sim_title",
    "sim_word",
    "sim_sent",
    "sim_emocao",
]


(
    categoria_id_weight,
    word_cluster_weight,
    NMF_topic_weight,
    sent_cluster_weight,
    BERT_topic_weight,
    sim_title_weight,
    sim_word_weight,
    sim_sent_weight,
    sim_emocao_weight,
) = (0, 0, 0, 0, 0, 0, 0, 0, 0)
col_widths = [3, 1]

with st.container():
    col_left, col_right = st.columns(col_widths)
    with col_left:
        st.markdown("#### Coleção / Categoria")
        st.write("Indica se os hinos pertencem à mesma coletânea ou tema.")
    with col_right:
        categoria_id_weight = col_right.slider(
            label="Coleção / Categoria",
            label_visibility="hidden",
            min_value=0,
            max_value=100,
            value=100,
            step=1,
            key="categoria_id",
        )


with st.container():
    col_left, col_right = st.columns(col_widths)
    with col_left:
        st.markdown("#### Similaridade de títulos")
        st.write(
            "Avalia a similaridade entre títulos dos hinos usando *token set ratio*."
        )
    with col_right:
        sim_title_weight = col_right.slider(
            label="Similaridade de títulos",
            label_visibility="hidden",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            key="sim_title",
        )

with st.container():
    col_left, col_right = st.columns(col_widths)
    with col_left:
        st.markdown("#### Similaridade lexical (Word Embeddings)")
        st.write(
            "Avalia a proximidade semântica entre os hinos com base em embeddings de palavras."
        )
    with col_right:
        sim_word_weight = col_right.slider(
            label="Similaridade lexical (Word Embeddings)",
            label_visibility="hidden",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            key="sim_word",
        )


with st.container():
    col_left, col_right = st.columns(col_widths)
    with col_left:
        st.markdown("#### Agrupamento lexical")
        st.write(
            "Compara se os hinos pertencem ao mesmo cluster de palavras (análise de agrupamento)."
        )
    with col_right:
        word_cluster_weight = col_right.slider(
            label="Agrupamento lexical",
            label_visibility="hidden",
            min_value=0,
            max_value=100,
            value=90,
            step=1,
            key="word_cluster",
        )


with st.container():
    col_left, col_right = st.columns(col_widths)
    with col_left:
        st.markdown("#### Tópico (NMF)")
        st.write("Considera similaridade de tópicos extraídos por NMF.")
    with col_right:
        NMF_topic_weight = col_right.slider(
            label="Tópico (NMF)",
            label_visibility="hidden",
            min_value=0,
            max_value=100,
            value=80,
            step=1,
            key="NMF_topic",
        )


with st.container():
    col_left, col_right = st.columns(col_widths)
    with col_left:
        st.markdown("#### Similaridade de sentenças (Sentence Embeddings)")
        st.write(
            "Avalia a proximidade semântica entre representações de sentenças dos hinos."
        )
    with col_right:
        sim_sent_weight = col_right.slider(
            label="Similaridade de sentenças (Sentence Embeddings)",
            label_visibility="hidden",
            min_value=0,
            max_value=100,
            value=40,
            step=1,
            key="sim_sent",
        )


with st.container():
    col_left, col_right = st.columns(col_widths)
    with col_left:
        st.markdown("#### Agrupamento de sentenças")
        st.write("Compara se os hinos pertencem ao mesmo cluster baseado em sentenças.")
    with col_right:
        sent_cluster_weight = col_right.slider(
            label="Agrupamento de sentenças",
            label_visibility="hidden",
            min_value=0,
            max_value=100,
            value=70,
            step=1,
            key="sent_cluster",
        )


with st.container():
    col_left, col_right = st.columns(col_widths)
    with col_left:
        st.markdown("#### Tópico (BERT)")
        st.write(
            "Considera correspondência de tópicos extraídos via modelos baseados em BERT."
        )
    with col_right:
        BERT_topic_weight = col_right.slider(
            label="Tópico (BERT)",
            label_visibility="hidden",
            min_value=0,
            max_value=100,
            value=60,
            step=1,
            key="BERT_topic",
        )


with st.container():
    col_left, col_right = st.columns(col_widths)
    with col_left:
        st.markdown("#### Similaridade emocional")
        st.write(
            "Compara perfis emocionais derivados da análise de emoções no texto dos hinos."
        )
    with col_right:
        sim_emocao_weight = col_right.slider(
            label="Similaridade emocional",
            label_visibility="hidden",
            min_value=0,
            max_value=100,
            value=30,
            step=1,
            key="sim_emocao",
        )

weights = [
    categoria_id_weight,
    word_cluster_weight,
    NMF_topic_weight,
    sent_cluster_weight,
    BERT_topic_weight,
    sim_title_weight,
    sim_word_weight,
    sim_sent_weight,
    sim_emocao_weight,
]


st.divider()


# - Escolha do hino
"""## Escolha um hino para ver sugestões similares:

Considerando os pesos definidos acima, selecione um hino para o qual deseja encontrar 
sugestões similares.
"""
hinos_opcoes = [f"{num} - {row['nome']}" for num, row in hinos_analise.iterrows()]
hino_selecionado = st.selectbox(
    "Pesquisar hino (número ou nome)",
    options=hinos_opcoes,
    placeholder="Digite para buscar...",
    index=None,
    help="Digite o número ou parte do nome do hino para pesquisar",
)
if not hino_selecionado:
    st.warning("Por favor, selecione um hino para prosseguir.")
    st.stop()

hymn_num = int(hino_selecionado.split(" - ")[0])
hino_sample = hinos_analise.loc[hymn_num]


hinos_restantes = hinos_analise[hinos_analise.index != hino_sample.name].copy()
# se categoria_id é a mesma de sample, então 1, senão 0
hinos_restantes["categoria_id"] = (
    hinos_restantes["categoria_id"] == hino_sample["categoria_id"]
).astype(int)
hinos_restantes["word_cluster"] = (
    hinos_restantes["word_cluster"] == hino_sample["word_cluster"]
).astype(int)
hinos_restantes["NMF_topic"] = (
    hinos_restantes["NMF_topic"] == hino_sample["NMF_topic"]
).astype(int)
hinos_restantes["sent_cluster"] = (
    hinos_restantes["sent_cluster"] == hino_sample["sent_cluster"]
).astype(int)
hinos_restantes["BERT_topic"] = (
    hinos_restantes["BERT_topic"] == hino_sample["BERT_topic"]
).astype(int)
hinos_restantes = hinos_restantes[
    [
        "nome",
        "texto_limpo",
        "categoria_id",
        "word_cluster",
        "NMF_topic",
        "sent_cluster",
        "BERT_topic",
    ]
]

similarity_matrix_titles_sample = similarity_titles.loc[
    hino_sample.name, hinos_restantes.index
]
similarity_matrix_words_sample = similarity_word.loc[
    hino_sample.name, hinos_restantes.index
]
similarity_matrix_sent_sample = similarity_sent.loc[
    hino_sample.name, hinos_restantes.index
]
similarity_matrix_emocoes_sample = similarity_emocoes.loc[
    hino_sample.name, hinos_restantes.index
]
hinos_restantes["sim_title"] = similarity_matrix_titles_sample.values
hinos_restantes["sim_word"] = similarity_matrix_words_sample.values
hinos_restantes["sim_sent"] = similarity_matrix_sent_sample.values
hinos_restantes["sim_emocao"] = similarity_matrix_emocoes_sample.values


st.divider()

# - Resultados
"""
## Resultados da seleção TOPSIS

"""

# Valida se os sliders já retornaram valores (não são None)
if any(w is None for w in weights):
    st.warning(
        "Aguarde a inicialização dos sliders ou ajuste-os para prosseguir com a seleção TOPSIS."
    )
    st.stop()

# Evita divisão por zero: exige soma > 0
sum_w = sum(weights)
if sum_w == 0:
    st.warning(
        "A soma dos pesos é zero. Ajuste pelo menos um peso para um valor maior que zero."
    )
    st.stop()

# Normaliza os pesos
weights = [w / sum_w for w in weights]
if sum(weights) != 1:
    st.warning(
        "Aguarde a inicialização dos sliders ou ajuste-os para prosseguir com a seleção TOPSIS."
    )
    st.stop()

profit_cost = [1, 1, 1, 1, 1, 1, 1, 1, 1]


# executa TOPSIS apenas após validações
# valida e prepara a matriz de critérios para evitar None dentro da função TOPSIS
X = hinos_restantes[categories].copy()
# força tipos numéricos; se houver valores não-convertíveis vira NaN -> substitui por 0
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

# valida consistência de dimensões com os vetores de peso e profit_cost
if X.shape[1] != len(weights) or X.shape[1] != len(profit_cost):
    st.error(
        f"Número de critérios ({X.shape[1]}) diferente do tamanho de weights ({len(weights)}) "
        f"ou profit_cost ({len(profit_cost)}). Ajuste antes de prosseguir."
    )
    st.stop()

output = TOPSIS(
    X,
    weights,
    profit_cost,
)
scores = pd.DataFrame(output)
scores.columns = ["topsis_score"]
hinos_restantes = pd.concat([hinos_restantes, scores], axis=1).sort_values(
    by="topsis_score", ascending=False
)

st.markdown(f"Sugestões para o hino: **{hino_sample.name} - {hino_sample['nome']}**")
st.dataframe(
    hinos_restantes[["nome", "topsis_score"]].head(20).rename_axis("Nº"),
    width="stretch",
    column_config={
        "nome": st.column_config.TextColumn("Nome"),
        "topsis_score": st.column_config.NumberColumn(
            "TOPSIS Score",
            format="%.2f",
        ),
    },
    height=500,
)
