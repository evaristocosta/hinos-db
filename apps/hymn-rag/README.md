# 🎵 Busca Inteligente de Hinos

Sistema de busca inteligente de hinos usando RAG (Retrieval-Augmented Generation) com Hugging Face.

## 📋 Sobre o Projeto

Esta aplicação utiliza técnicas avançadas de busca semântica e geração de texto para encontrar hinos da Coletânea da Igreja Cristã Maranata de forma inteligente. O sistema:

- Realiza busca semântica usando embeddings
- Filtra por categorias e coletâneas
- Gera respostas contextualizadas usando LLM
- Apresenta referências bíblicas relacionadas
- Interface amigável e responsiva

## 🚀 Deploy no Streamlit Community Cloud

### Pré-requisitos

1. Conta no [GitHub](https://github.com)
2. Conta no [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Token de API do [Hugging Face](https://huggingface.co/settings/tokens)
4. Repositório Git com este projeto

### Passos para Deploy

1. **Suba o código para o GitHub**

   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/seu-usuario/seu-repositorio.git
   git push -u origin main
   ```

2. **Obtenha seu Token do Hugging Face**

   - Acesse https://huggingface.co/settings/tokens
   - Crie um novo token (ou use um existente)
   - Copie o token para usar no próximo passo

3. **Acesse o Streamlit Community Cloud**

   - Vá para https://share.streamlit.io/
   - Faça login com sua conta GitHub

4. **Crie um novo app**

   - Clique em "New app"
   - Selecione seu repositório
   - Branch: `main` (ou sua branch principal)
   - Main file path: `apps/hymn-rag/streamlit_app.py`
   - App URL: escolha um nome personalizado (opcional)

5. **Configure os Secrets**

   - Na página de configuração do app, vá para "Advanced settings"
   - Em "Secrets", adicione:

   ```toml
   HUGGINGFACE_API_TOKEN = "seu_token_aqui"
   ```

6. **Aguarde o deploy**
   - O Streamlit Cloud irá instalar as dependências automaticamente
   - O processo pode levar alguns minutos na primeira vez

### ⚠️ Notas Importantes

- **Banco de dados**: A aplicação usa o banco SQLite que deve estar em `apps/shared/assets/hinos.db`
- **Vectorstore**: Os embeddings pré-calculados devem estar em `apps/shared/assets/hymn_rag_vectorstore/`
- **Chunks**: Os chunks pré-processados devem estar em `apps/shared/assets/hymn_rag_chunks.pkl`
- **Token Hugging Face**: É obrigatório para o funcionamento do sistema RAG
- **Limitações**: O plano gratuito do Streamlit Cloud tem limites de recursos e tempo de execução

### 🔧 Estrutura de Arquivos Necessária

```
apps/hymn-rag/
├── streamlit_app.py        # Aplicação principal
├── rag_hf.py              # Sistema RAG com Hugging Face
├── requirements.txt       # Dependências Python
├── packages.txt          # Dependências do sistema
└── README.md            # Este arquivo

apps/shared/assets/
├── hinos.db                              # Banco de dados SQLite
├── hymn_rag_vectorstore/                 # Vectorstore Chroma
└── hymn_rag_chunks.pkl                   # Chunks pré-processados
```

## 💻 Desenvolvimento Local

Para rodar localmente:

1. Clone o repositório
2. Crie um arquivo `.env` com:
   ```
   HUGGINGFACE_API_TOKEN=seu_token_aqui
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. Execute o Streamlit:
   ```bash
   streamlit run streamlit_app.py
   ```

## 🛠️ Tecnologias Utilizadas

- **Streamlit**: Interface web
- **Hugging Face**: Modelos de embeddings e LLM
- **LangChain**: Framework RAG
- **ChromaDB**: Vectorstore para busca semântica
- **SQLite**: Banco de dados de hinos

## 📝 Licença

Este projeto é parte da Coletânea de Hinos da Igreja Cristã Maranata.

---

<div style="text-align: center; color: #666; font-size: 0.9rem;">
    Desenvolvido com ❤️ usando Streamlit e Hugging Face
</div>
