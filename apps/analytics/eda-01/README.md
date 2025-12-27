# 🎵 Hinos em Dados - EDA da Coletânea

Análise Exploratória de Dados (EDA) da Coletânea de Hinos da Igreja Cristã Maranata.

## 📋 Sobre o Projeto

Este projeto realiza uma análise exploratória completa dos hinos utilizando técnicas de Ciência de Dados e Processamento de Linguagem Natural (NLP) para:

- Compreender padrões nos títulos e letras dos hinos
- Identificar categorias temáticas e características dos louvores
- Analisar emoções presentes nas letras
- Explorar similaridades entre os hinos
- Fornecer insights sobre o conteúdo da coletânea

## 🚀 Deploy no Streamlit Community Cloud

### Pré-requisitos

1. Conta no [GitHub](https://github.com)
2. Conta no [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Repositório Git com este projeto

### Passos para Deploy

1. **Suba o código para o GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/seu-usuario/seu-repositorio.git
   git push -u origin main
   ```

2. **Acesse o Streamlit Community Cloud**
   - Vá para https://share.streamlit.io/
   - Faça login com sua conta GitHub

3. **Crie um novo app**
   - Clique em "New app"
   - Selecione seu repositório
   - Branch: `main` (ou sua branch principal)
   - Main file path: `eda1/streamlit_app.py`
   - App URL: escolha um nome personalizado (opcional)

4. **Aguarde o deploy**
   - O Streamlit Cloud irá instalar as dependências automaticamente
   - O processo pode levar alguns minutos

### ⚠️ Notas Importantes

- **Tamanho dos arquivos**: Os arquivos `.pkl` e `.db` na pasta `assets/` somam aproximadamente 35MB. Certifique-se de que estão no repositório e não foram filtrados pelo `.gitignore`.

- **Limite do GitHub**: Se os arquivos forem muito grandes (> 100MB individualmente), você pode precisar usar [Git LFS (Large File Storage)](https://git-lfs.github.com/).

- **NLTK Data**: Se houver erros relacionados ao NLTK, pode ser necessário adicionar um arquivo `nltk.txt` na raiz com:
  ```
  stopwords
  ```

## 📁 Estrutura do Projeto

```
eda1/
├── .streamlit/
│   └── config.toml          # Configurações de tema do Streamlit
├── assets/                   # Dados e arquivos processados
│   ├── database.db          # Banco de dados SQLite
│   ├── *.pkl                # Arquivos pickle com dados processados
│   ├── stopwords-br.txt     # Stop words em português
│   └── wordcloud.png        # Imagem da nuvem de palavras
├── src/                      # Código fonte das páginas
│   ├── main.py              # Página inicial
│   ├── pipeline.py          # Funções de carregamento de dados
│   ├── tabela.py            # Tabela exploratória
│   ├── categorias.py        # Análise de categorias
│   ├── analise_titulo.py    # Análise de títulos
│   ├── analise_texto.py     # Exploração de palavras
│   ├── analise_palavras.py  # Análise de palavras
│   ├── word_embeddings.py   # Embeddings de palavras
│   ├── sent_embeddings.py   # Embeddings de frases
│   ├── emocoes.py           # Análise de emoções
│   └── topsis_escolha.py    # Seleção de similares
├── notebooks/                # Notebooks Jupyter (desenvolvimento)
├── streamlit_app.py         # Arquivo principal do Streamlit
├── requirements.txt         # Dependências Python
├── .gitignore               # Arquivos ignorados pelo Git
└── README.md                # Este arquivo

```

## 🛠️ Executar Localmente

1. Clone o repositório
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o app:
   ```bash
   streamlit run streamlit_app.py
   ```

## 👨‍💻 Autor

**Lucas Piccioni Costa**
- 📧 Email: lucascosta74@gmail.com
- 📸 Instagram: [@lucas.costa74](https://www.instagram.com/lucas.costa74/)
- 💼 LinkedIn: [lucascosta74](https://www.linkedin.com/in/lucascosta74/)
- 🐙 GitHub: [evaristocosta](https://github.com/evaristocosta)

## 📄 Licença

Este projeto é de código aberto e está disponível sob a licença MIT.
