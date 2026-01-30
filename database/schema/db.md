# Documentação do Database - Hinos DB

## 📁 Estrutura da Pasta `database`

A pasta `database` contém toda a estrutura e ferramentas para gerenciar o banco de dados SQLite do projeto Hinos DB. A organização é a seguinte:

```
database/
├── migrations/         # Scripts SQL de migração (versionados)
├── schema/             # Documentação e definição do schema
└── tools/              # Scripts auxiliares para gerenciar o banco
```

### 📂 `migrations/`

Contém os arquivos de migração SQL que constroem o banco de dados de forma incremental e versionada. Cada arquivo segue o padrão de nomenclatura `###_descricao.sql` onde:

- `###`: Número sequencial de três dígitos (001, 002, 003, ...)
- `descricao`: Descrição breve da alteração

**Migrações atuais:**

- `001_create_main_tables.sql` - Cria as tabelas principais do banco
- `002_add_coletaneas.sql` - Adiciona as coletâneas iniciais
- `003-006_*.sql` - Importa hinos de diferentes coletâneas específicas
- `007_add_categorias.sql` - Adiciona categorias dos hinos
- `008_alter_categoria_hinos_coletanea_normal.sql` - Ajusta categorização
- `009_fix_hinos.sql` - Correções nos dados de hinos
- `010_add_hinos_avulsos.sql` - Adiciona hinos avulsos

> **Importante:** As migrações são executadas em ordem numérica. Nunca modifique migrações já aplicadas. Para fazer alterações, crie uma nova migração.

### 📂 `schema/`

Contém a documentação e definições do schema do banco de dados em diferentes formatos:

- **`db-overview.*`** - Visão conceitual (alto nível) do modelo de dados em DBML e PlantUML, gerada manualmente e atualizada conforme o modelo evolui.
- **`db-schema.*`** - Definição técnica (baixo nível) mais atual do schema em SQL puro e PlantUML, gerada automaticamente a partir do banco de dados.

### 📂 `tools/`

Ferramentas Python para gerenciar e manter o banco de dados:

- **`run_migrations.py`** - Script para executar as migrações
- **`generate_schema_puml.py`** - Gera o diagrama PlantUML do schema
- **`generate_schema_sql.py`** - Gera o arquivo SQL do schema

---

## 🚀 Como Rodar as Migrações

### Pré-requisitos

1. **SQLite3** instalado e disponível no PATH do sistema
2. **Python 3.x** (se for usar o script de migração)

### Método 1: Usando o Script Python (Recomendado)

O script `run_migrations.py` automatiza o processo de execução de todas as migrações em ordem:

```bash
cd database/tools
python run_migrations.py
```

> ⚠️ **Atenção:** Este script **deleta o banco existente** antes de rodar as migrações. Use com cuidado em ambientes de produção!

### Método 2: Manual via SQLite3

Para executar as migrações manualmente:

```bash
# Navegue até a pasta database
cd database

# Execute cada migração em ordem
sqlite3 database.db < migrations/001_create_main_tables.sql
sqlite3 database.db < migrations/002_add_coletaneas.sql
# ... e assim por diante
```

---

## 🔄 Fluxo de Trabalho

### Adicionando uma Nova Migração

1. Crie um novo arquivo na pasta `migrations/` com o próximo número sequencial:

   ```
   011_sua_descricao_aqui.sql
   ```

2. Escreva os comandos SQL necessários (CREATE, ALTER, INSERT, etc.)

3. Execute as migrações para testar:

   ```bash
   cd database/tools
   python run_migrations.py
   ```

4. Se necessário, atualize os arquivos de schema:
   - `db-overview.*` - Se alterou o modelo de dados

### Atualizando o Schema

Após modificações significativas:

```bash
cd database/tools
python generate_schema_sql.py
python generate_schema_puml.py
```

---

## 📋 Modelo de Dados Principal

O banco contém as seguintes tabelas principais:

- **`autor`** - Autores de hinos (compositores, letristas)
- **`autor_acao`** - Tipos de ação do autor (compôs, traduziu, adaptou)
- **`categoria`** - Categorias de hinos
- **`coletanea`** - Coletâneas/livros de hinos
- **`hino`** - Hinos propriamente ditos (tabela central)
- **`hino_autor`** - Relacionamento entre hinos e autores
