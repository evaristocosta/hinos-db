#!/usr/bin/env python
"""
Aplicação Streamlit para busca de hinos usando RAG
"""
import streamlit as st
from pathlib import Path
import sys

# Adiciona o diretório do projeto ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_hf import HymnRAG

# Configuração da página
st.set_page_config(
    page_title="Busca Inteligente de Hinos",
    page_icon="🎵",
    layout="wide",
    # initial_sidebar_state="expanded",
)

# CSS customizado
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #818d3f;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #818d3f;
        color: white;
    }
    .hymn-result {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    div[data-testid="InputInstructions"] > span:nth-child(1) {
        visibility: hidden;
    }
    div[data-testid="stVerticalBlock"] {
        justify-content: end;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown(
    '<div class="main-header">🎵 Busca Inteligente de Hinos</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Sistema para consulta de hinos da Igreja Cristã Maranata</div>',
    unsafe_allow_html=True,
)


# Inicialização do sistema RAG
@st.cache_resource(
    show_spinner="Carregando sistema...", ttl=3600
)  # Cache expira em 1 hora
def load_rag():
    """Carrega o sistema RAG uma única vez"""
    try:
        is_development = st.secrets.get("DEVELOPMENT_MODE", False)
        return HymnRAG(verbose=is_development)
    except Exception as e:
        st.error(f"❌ Erro ao inicializar o sistema: {str(e)}")
        st.stop()


rag = load_rag()

# Sidebar - Filtros
with st.sidebar:
    # st.header("Configurações")

    st.subheader("🔍 Filtros")

    # Checkbox para habilitar filtros automáticos
    # auto_filters = st.checkbox(
    #     "Extrair filtros automaticamente da consulta",
    #     value=False,
    #     help="Detecta automaticamente categorias e coletâneas mencionadas na consulta",
    # )

    # Filtros manuais
    # Categorias
    categorias_disponiveis = list(rag.categorias.keys())
    categorias_selecionadas = st.multiselect(
        "Categorias",
        options=categorias_disponiveis,
        help="Selecione uma ou mais categorias para filtrar",
        placeholder="Selecione uma ou mais categorias",
    )

    # Coletâneas
    coletaneas_disponiveis = list(rag.coletaneas.keys())
    coletaneas_selecionadas = st.multiselect(
        "Coletâneas",
        options=coletaneas_disponiveis,
        help="Selecione uma ou mais coletâneas para filtrar",
        placeholder="Selecione uma ou mais coletâneas",
    )

    # Informações
    # st.markdown("---")
    # st.subheader("ℹ️ Sobre")
    # st.info(
    #     f"""
    # **Total de hinos:** {rag.total_hinos}

    # **Categorias:** {len(rag.categorias)}

    # **Coletâneas:** {len(rag.coletaneas)}
    # """
    # )

    # Exemplos
    st.markdown("---")
    st.subheader("💡 Exemplos de consulta")
    st.markdown(
        """
    - Hinos sobre unidade
    - Louvores para visita em lar de idosos
    - Hinos que combinam com Isaías 43:2
    - Músicas sobre a volta de Jesus
    - Hinos de consolo
    """
    )

# Área principal
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "Digite sua consulta:",
        placeholder="Ex: Hinos sobre graça e salvação",
        help="Digite palavras-chave, temas ou referências bíblicas",
    )

with col2:
    search_button = st.button("🔍 Buscar", type="primary", use_container_width=True)

st.expander("ℹ️ Instruções", expanded=False).markdown(
    """
Imagine que você está fazendo uma pergunta a um assistente virtual que tem acesso aos hinos da Igreja Cristã
Maranata, inclusive de crianças e avulsos. A resposta será de sugestões de hinos que melhor respondem à sua consulta.
Importante:
- Se quiser usar uma referência bíblica, não precisa colocar o texto inteiro, apenas a referência (ex: João 3:16);
- Você pode aplicar filtros manuais na barra lateral, escolhendo categorias e coletâneas específicas.
"""
)

# Área de resultados
if search_button:
    if not query:
        st.warning("⚠️ Por favor, digite uma consulta.")
    else:
        with st.spinner("🔎 Buscando hinos..."):
            try:
                # Prepara filtros manuais
                manual_categorias = (
                    categorias_selecionadas if categorias_selecionadas else None
                )
                manual_coletaneas = (
                    coletaneas_selecionadas if coletaneas_selecionadas else None
                )

                # Executa a consulta e obtém docs e generator
                docs, bible_context, response_stream = rag.query_stream(
                    question=query,
                    auto_filters=False,
                    manual_categorias=manual_categorias,
                    manual_coletaneas=manual_coletaneas,
                )

                # Exibe resultado
                st.markdown("---")
                st.subheader("📋 Resultado")

                # Container para o resultado com streaming
                with st.container():
                    st.write_stream(response_stream)

                # Expander com texto bíblico de contexto, se houver
                if bible_context:
                    with st.expander("📖 Contexto Bíblico Utilizado"):
                        st.markdown(bible_context)

                # Expander com hinos encontrados
                if docs:
                    with st.expander(f"🎵 Hinos relacionados ({len(docs)})"):
                        st.write(
                            "Estes são alguns dos hinos que podem ter relação com sua consulta:"
                        )
                        for i, doc in enumerate(docs, start=1):
                            numero = doc.metadata.get("numero", "N/A")
                            if numero == "null":
                                numero = "N/A"
                            nome = doc.metadata.get("nome", "Sem título")
                            categoria_id = doc.metadata.get("categoria_id", None)
                            coletanea_id = doc.metadata.get("coletanea_id", None)

                            st.markdown(f"{i}. [{numero}] {nome}")

                            # Tags coloridas lado a lado
                            tags = []
                            if categoria_id:
                                # Busca a categoria pelo id (valor no dict)
                                categoria_nome = next(
                                    (
                                        k
                                        for k, v in rag.categorias.items()
                                        if v == categoria_id
                                    ),
                                    "Desconhecida",
                                )
                                tags.append(
                                    f'<span style="background-color: #1565c0; color: #ffffff; padding: 3px 10px; border-radius: 12px; font-size: 0.85rem; margin-right: 8px; display: inline-block;">{categoria_nome.title()}</span>'
                                )

                            if coletanea_id:
                                # Busca a coletânea pelo id (valor no dict)
                                coletanea_nome = next(
                                    (
                                        k
                                        for k, v in rag.coletaneas.items()
                                        if v == coletanea_id
                                    ),
                                    "Desconhecida",
                                )
                                tags.append(
                                    f'<span style="background-color: #212510; color: #ffffff; padding: 3px 10px; border-radius: 12px; font-size: 0.85rem; margin-right: 8px; display: inline-block;">{coletanea_nome.title()}</span>'
                                )

                            if tags:
                                st.markdown("".join(tags), unsafe_allow_html=True)

                # Informação sobre filtros aplicados
                if manual_categorias or manual_coletaneas:  # or auto_filters:
                    with st.expander("ℹ️ Filtros Aplicados"):
                        if manual_categorias:
                            st.write(f"**Categorias:** {', '.join(manual_categorias)}")
                        if manual_coletaneas:
                            st.write(f"**Coletâneas:** {', '.join(manual_coletaneas)}")
                        # if auto_filters:
                        #     st.write("**Filtros automáticos:** Habilitados")

            except Exception as e:
                st.error(f"❌ Erro ao processar consulta: {str(e)}")
                st.exception(e)


# Footer
st.markdown("---")
# Disclaimer
st.markdown(
    """
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    ⚠️ Este sistema é uma ferramenta de auxílio para consulta de hinos. 
    É um trabalho voluntário, e não possui afiliação oficial com a Igreja Cristã Maranata.
    As respostas são geradas automaticamente e podem não refletir todas as nuances dos hinos disponíveis. 
    Sempre consulte a coletânea e canais oficiais para informações completas.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("&#8203;")  # Espaço vertical pequeno

# Feedback
st.markdown(
    """
<div style="text-align: center; color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">
    💬 Sugestões, dúvidas ou problemas? Entre em contato:<br>
    📧 <a href="mailto:lucascosta74@gmail.com">lucascosta74@gmail.com</a> | 
    📱 <a href="https://instagram.com/lucas.costa74" target="_blank">@lucas.costa74</a>
</div>
""",
    unsafe_allow_html=True,
)

# Desenvolvimento
st.markdown(
    """
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    Desenvolvido com ❤️ por <a href="https://github.com/evaristocosta" target="_blank">Lucas Costa</a> 
    usando Streamlit, <a href="https://www.abibliadigital.com.br/" target="_blank">ABibliaDigital</a> e Hugging Face
</div>
""",
    unsafe_allow_html=True,
)
