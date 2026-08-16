-- Migration 012: Remove "numero" and "coletanea_id" columns from "hino",
-- and add foreign key constraints to "hino_coletanea"

PRAGMA foreign_keys = OFF;

-- ============================================
-- 1. Recreate "hino" without "numero" and "coletanea_id"
-- ============================================

CREATE TABLE IF NOT EXISTS "hino_new" (
  "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "nome" TEXT NOT NULL,
  "nome_pt" TEXT,
  "idioma" TEXT,
  "texto" TEXT,
  "texto_processado" TEXT,
  "tom" TEXT,
  "texto_cifra" TEXT,
  "cifra" TEXT,
  "categoria_id" INTEGER,
  "ano_composicao" INTEGER,
  "date_insert" DATETIME NOT NULL,
  "date_update" DATETIME NOT NULL,
  FOREIGN KEY ("categoria_id") REFERENCES "categoria" ("id")
);

INSERT INTO "hino_new" (
    "id", "nome", "nome_pt", "idioma", "texto", "texto_processado",
    "tom", "texto_cifra", "cifra", "categoria_id", "ano_composicao",
    "date_insert", "date_update"
)
SELECT
    "id", "nome", "nome_pt", "idioma", "texto", "texto_processado",
    "tom", "texto_cifra", "cifra", "categoria_id", "ano_composicao",
    "date_insert", "date_update"
FROM "hino";

DROP TABLE "hino";

ALTER TABLE "hino_new" RENAME TO "hino";

-- ============================================
-- 2. Recreate "hino_coletanea" with foreign keys
-- ============================================

CREATE TABLE IF NOT EXISTS "hino_coletanea_new" (
  "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "hino_id" INTEGER NOT NULL,
  "hino_numero" TEXT,
  "coletanea_id" INTEGER NOT NULL,
  FOREIGN KEY ("hino_id") REFERENCES "hino" ("id"),
  FOREIGN KEY ("coletanea_id") REFERENCES "coletanea" ("id")
);

INSERT INTO "hino_coletanea_new" ("id", "hino_id", "hino_numero", "coletanea_id")
SELECT "id", "hino_id", "hino_numero", "coletanea_id"
FROM "hino_coletanea";

DROP TABLE "hino_coletanea";

ALTER TABLE "hino_coletanea_new" RENAME TO "hino_coletanea";

PRAGMA foreign_keys = ON;