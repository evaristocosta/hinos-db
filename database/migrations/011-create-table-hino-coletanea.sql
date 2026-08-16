CREATE TABLE IF NOT EXISTS "hino_coletanea" (
  "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "hino_id" INTEGER NOT NULL,
  "hino_numero" TEXT,
  "coletanea_id" INTEGER NOT NULL
);

INSERT INTO "hino_coletanea" ("hino_id", "hino_numero", "coletanea_id")
SELECT id AS hino_id, numero AS hino_numero, coletanea_id FROM hino;
