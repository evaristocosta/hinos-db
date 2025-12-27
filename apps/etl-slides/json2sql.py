import logging
import json
import re
import glob
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


erros_conhecidos = [
    "ABENÇOA-NOS SENHOR, DERRAMA SOBRE NÓS TUA PAZ ABENÇOA-NOS SENHOR, DERRAMA SOBRE NÓS TEU AMOR.",  # indice extra
    "MEU JESUS, SALVADOR,  OUTRO IGUAL NÃO HÁ. TODOS OS DIAS QUERO LOUVAR  AS MARAVILHAS DE TEU AMOR. CONSOLO, ABRIGO,  FORÇA E REFÚGIO É O SENHOR. COM TODO O MEU SER,  COM TUDO O QUE SOU,  SEMPRE TE ADORAREI.",  # pedaco de louvor
    "PODES CLAMAR, PODES CHORAR, EU ESTAREI PRONTO PRA TE AJUDAR, E SE O CORAÇÃO DESFALECER, CONFIA EM MIM SOU JESUS  E TE FAÇO VENCER.",  # pedaco de louvor
    "AO CORDEIRO GLÓRIA E HONRA,  SALVOS NÃO CESSEIS DE DAR. GLÓRIA, HONRA SEMPRE A DEUS ENTOAREI!  AMÉM!",  # indice extra
]
TAGS_LITERAIS = [
    "TODOS",
    "M",
    "H",
    "T",
    "BIS",
    "VARÕES",
    "SERVAS",
]
TAGS_CONTROLE = [
    "ÍNDICE",
    "CORO (2X)",
    "\n\nCORO",
    "CORO\n",
    "1X",
    "2X",
    "3X",
    "4X",
    "()",
    "(TODOS)",
    "(M)",
    "(H)",
    "(T)",
    "(BIS)",
    "(VARÕES)",
    "(SERVAS)",
    "REPETIR O LOUVOR",
    "REPETIR 1ª ESTROFE",
    "REPETIR A 1ª ESTROFE",
    "REPETIR 2ª ESTROFE",
    "REPETIR A 2ª ESTROFE",
    "REPETIR ESTROFE",
    "REPETIR A ESTROFE",
    "FINAL:",
    "BIS NO FINAL",
    "IGREJA CRISTÃ MARANATA",
    "ATUALIZAÇÃO",
    "\nINSTRUMENTOS",
]


def get_texts(d):
    if isinstance(d, dict):
        for k, v in d.items():
            if k == "text":
                yield v
            else:
                yield from get_texts(v)
    elif isinstance(d, list):
        for v in d:
            yield from get_texts(v)


def return_possible_title(texts):
    possible_title = [
        text
        for text in texts
        if all(opt not in text.upper() for opt in TAGS_CONTROLE)
        and text.strip() != ""
        and text.upper() not in TAGS_LITERAIS
    ]
    if possible_title:
        min_string = min(possible_title, key=len)
        title_index = texts.index(min_string)
        title = min_string.upper().replace("\n", " ")
        if title not in erros_conhecidos:
            return title, title_index
    return None, None


def set_text_clean(texts_wo_title):
    texts_clean = [re.sub(r"[ \t]{2,}", " ", text) for text in texts_wo_title]
    texts_clean = [line for line in texts_clean if line not in TAGS_LITERAIS]
    new_texts_clean = []
    for line in texts_clean:
        for tag in TAGS_CONTROLE:
            line = line.upper().replace(tag, "")
        line = line.strip()
        if line:
            new_texts_clean.append(line)
    texts_clean = new_texts_clean
    coro_regex = re.compile(r"^CORO\s*")
    texts_clean = [line for line in texts_clean if not coro_regex.match(line)]
    texts_clean = " ".join(texts_clean)
    texts_clean = texts_clean.replace("\n", " ")
    texts_clean = re.sub(r"[ \t]{2,}", " ", texts_clean)
    # remove all double quotes
    texts_clean = texts_clean.replace("“", "")
    texts_clean = texts_clean.replace("”", "")
    texts_clean = texts_clean.replace('"', "")
    return texts_clean


def set_text_full(texts_wo_title):
    texts_full = [
        line
        for line in texts_wo_title
        if line.upper() != "ÍNDICE" and line.strip() != ""
    ]
    texts_full = "\n\n".join(texts_full)
    texts_full = texts_full.replace("\n\nBIS", "\nBIS")
    texts_full = texts_full.replace('"', "'")
    return texts_full


def return_number(title):
    if title:
        for word in title.split(" "):
            match = re.search(r"\d+", word)
            if match:
                numero = match.group()
                # remove number from title
                title_clean = title.replace(match.group(), "").strip()
                # check if string starts with hifen and remove it
                if title_clean.startswith("-"):
                    title_clean = title_clean[1:].strip()
                else:
                    title_clean = title_clean.strip()

                return numero, title_clean
            else:
                return "null", title
    return "null", None


def process_structure(praises):
    new_structure = []
    for praise in tqdm(praises, desc="Processing praises", unit="praise"):
        texts = list(get_texts(praise))

        if not texts:
            continue

        # clean double or more spaces in string array
        texts = [text.replace("–", "-") for text in texts]

        title, title_index = return_possible_title(texts)
        numero, title_clean = return_number(title)

        texts_wo_title = texts.copy()
        if title_index is not None:
            texts_wo_title.pop(title_index)

        texts_full = set_text_full(texts_wo_title)
        texts_clean = set_text_clean(texts_wo_title)

        new_structure.append(
            {
                "numero": numero,
                "nome": title_clean if title_clean is not None else "null",
                "texto": texts_full,
                "texto_limpo": texts_clean,
            }
        )

    return new_structure


def json2sql(inicio: int = 3):
    logging.info("Starting json2sql conversion...")
    files_json = glob.glob("slides_json\\*.json")
    logging.info(f"Files found: {files_json}")

    for index, file in enumerate(files_json):
        logging.info(f"Processing file: {file}")
        file_name = (
            "00"
            + str(index + inicio)
            + "_add_"
            + file.split("\\")[1].split("pptx")[0].lower().replace(" ", "_")
            + "sql"
        )

        with open(file, "r", encoding="utf-8") as f:
            louvores = json.load(f)

        louvores_estruturados = process_structure(louvores)
        with open("..\\..\\database\\migrations\\" + file_name, "w", encoding="utf-8") as f:
            for hino in louvores_estruturados:
                text = (
                    "INSERT INTO hino (numero, nome, texto, texto_limpo, coletanea_id, date_insert, date_update) VALUES ('"
                    + hino["numero"]
                    + "', '"
                    + hino["nome"]
                    + "', '"
                    + hino["texto"].replace("\n", "\\n").replace("'", "''")
                    + "', '"
                    + hino["texto_limpo"].replace("'", "''")
                    + "', "
                    + str(index + 1)
                    + ", CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);\n"
                )
                f.write(text)


def main():
    json2sql()


if __name__ == "__main__":
    main()
