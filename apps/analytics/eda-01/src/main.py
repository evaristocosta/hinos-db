import streamlit as st
from pathlib import Path

st.title("🎵 Hinos em Dados")
"""
Seja bem-vindo ao **Hinos em Dados**!

Aqui você pode explorar diversas informações e análises estatísticas sobre os hinos da *Coletânea de 
Hinos da Igreja Cristã Maranata* (excluindo os hinos de Crianças, Intermediários e Adolescentes - CIAs).
"""

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_PATH = REPO_ROOT / "eda-01" / "assets" / "wordcloud.png"

st.image(ASSET_PATH, caption="Nuvem de palavras, da seção Exploração de Palavras")

"""
## 📊 Objetivo do Projeto

Este projeto tem como propósito realizar uma **Análise Exploratória de Dados (EDA)** da 
Coletânea, utilizando técnicas de Ciência de Dados e Processamento de Linguagem Natural (NLP) para:

- **Compreender padrões** nos títulos e letras dos hinos;
- **Identificar categorias temáticas** e características dos louvores;
- **Analisar emoções** presentes nas letras;
- **Explorar similaridades** entre os hinos usando diversas abordagens; 
- **Fornecer insights** sobre a riqueza do conteúdo da coletânea.

## 🛠️ Desenvolvimento

Todo o código-fonte e os notebooks Jupyter utilizados no desenvolvimento estão disponíveis no 
**GitHub** no repositório [evaristocosta/hinos_db](https://github.com/evaristocosta/hinos_db). 
Os notebooks de análise encontram-se na pasta `eda1/notebooks/`, onde você pode acompanhar 
passo a passo todo o processo de exploração e análise dos dados.

## 📋 Sumário

Utilize o menu lateral para navegar entre as diferentes análises disponíveis:
"""
st.info(
    "**Importante**: As análises estão em ordem de complexidade crescente.", icon="ℹ️"
)
"""
- **📆 Tabela Exploratória**: Visualize todos os hinos usados nesta análise em formato de tabela, com informações como título, 
categoria e texto do hino. É possível filtrar e buscar hinos específicos.

- **📑 Categorias dos Louvores**: Uma rápida visão geral da distribuição dos hinos por categorias da coletânea.

- **🔢 Tamanho dos Títulos**: Analise estatísticas sobre o comprimento e características dos títulos dos hinos.

- **🔡 Exploração de Palavras**: Descubra as palavras mais frequentes e padrões de vocabulário nas letras dos hinos.

- **✒️ Análise de Palavras**: Aprofunde-se na análise de palavras específicas e suas ocorrências ao longo da coletânea.

- **📝 Embeddings de Palavras**: Explore representações vetoriais de palavras e visualize similaridades semânticas.

- **🗒️ Embeddings de Frases**: Veja como frases completas dos hinos se relacionam semanticamente no espaço vetorial.

- **🎭 Análise de Emoções**: Descubra as emoções predominantes nas letras dos hinos através de análise de sentimentos.

- **✅ Seleção de Similares**: Use o método TOPSIS para encontrar hinos similares baseado em múltiplos critérios.

## 👨‍💻 Contato

Este projeto foi desenvolvido por **Lucas Piccioni Costa**. Se tiver alguma dúvida, sugestão ou quiser
conversar sobre o projeto, sinta-se à vontade para entrar em contato:

- 📧 Email: [lucascosta74@gmail.com](mailto:lucascosta74@gmail.com)
- 📸 Instagram: [lucas.costa74](https://www.instagram.com/lucas.costa74/)
- 💼 LinkedIn: [lucascosta74](https://www.linkedin.com/in/lucascosta74/)
- 🐙 GitHub: [evaristocosta](https://github.com/evaristocosta)


"""
