CREATE TABLE IF NOT EXISTS hino_de (
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  hino_id INTEGER NOT NULL,
  numero INTEGER,
  nome TEXT,
  nome_traduzido TEXT,
  texto TEXT,
  texto_limpo_traduzido TEXT
);

CREATE TABLE IF NOT EXISTS autor (
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  nome TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS coletanea (
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  nome TEXT NOT NULL,
  descricao TEXT,
  arquivo TEXT
);

CREATE TABLE IF NOT EXISTS autor_acao (
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  acao TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hino_autor (
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  hino_id INTEGER NOT NULL,
  autor_id INTEGER NOT NULL,
  autor_acao_id INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS hino (
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  coletanea_id INTEGER,
  categoria_id INTEGER,
  numero TEXT,
  nome TEXT NOT NULL,
  texto TEXT,
  texto_limpo TEXT,
  texto_estruturado TEXT,
  ano_composicao INTEGER,
  tom TEXT,
  date_insert DATETIME NOT NULL,
  date_update DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS categoria (
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  descricao TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hino_autor_fk (
  hino_autor_id INTEGER,
  autor_acao_id INTEGER,
  FOREIGN KEY (autor_acao_id) REFERENCES autor_acao (id),
  FOREIGN KEY (hino_autor_id) REFERENCES hino_autor (id)
);

CREATE TABLE IF NOT EXISTS hino_fk (
  hino_id INTEGER,
  coletanea_id INTEGER,
  categoria_id INTEGER,
  FOREIGN KEY (coletanea_id) REFERENCES coletanea (id),
  FOREIGN KEY (hino_id) REFERENCES hino_de (hino_id),
  FOREIGN KEY (categoria_id) REFERENCES categoria (id)
);