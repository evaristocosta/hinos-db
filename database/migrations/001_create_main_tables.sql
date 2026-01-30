CREATE TABLE IF NOT EXISTS "autor" (
  "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "nome" TEXT NOT NULL,
  "nacionalidade" TEXT
);

CREATE TABLE IF NOT EXISTS "autor_acao" (
  "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "acao" TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS "categoria" (
  "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "nome" TEXT NOT NULL,
  "descricao" TEXT
);

CREATE TABLE IF NOT EXISTS "coletanea" (
  "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "nome" TEXT NOT NULL,
  "apelido" TEXT NOT NULL,
  "descricao" TEXT,
  "arquivo" TEXT
);

CREATE TABLE IF NOT EXISTS "hino" (
  "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "numero" TEXT,
  "nome" TEXT NOT NULL,
  "nome_pt" TEXT,
  "idioma" TEXT,
  "texto" TEXT,
  "texto_limpo" TEXT,
  "texto_estruturado" TEXT,
  "tom" TEXT,
  "texto_cifra" TEXT,
  "cifra" TEXT,
  "coletanea_id" INTEGER,
  "categoria_id" INTEGER,
  "ano_composicao" INTEGER,
  "date_insert" DATETIME NOT NULL,
  "date_update" DATETIME NOT NULL,
  FOREIGN KEY ("categoria_id") REFERENCES "categoria" ("id"),
  FOREIGN KEY ("coletanea_id") REFERENCES "coletanea" ("id")
);

CREATE TABLE IF NOT EXISTS "hino_autor" (
  "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "hino_id" INTEGER NOT NULL,
  "autor_id" INTEGER NOT NULL,
  "autor_acao_id" INTEGER NOT NULL,
  FOREIGN KEY ("autor_id") REFERENCES "autor" ("id"),
  FOREIGN KEY ("hino_id") REFERENCES "hino" ("id"),
  FOREIGN KEY ("autor_acao_id") REFERENCES "autor_acao" ("id")
);