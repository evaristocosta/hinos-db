import logging
import glob
import re
import json
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def txt2json():
    logging.info("Starting txt2json conversion...")
    files_txt = glob.glob("slides_txt\\*.txt")
    logging.info(f"Files found: {files_txt}")

    auto_shape_pattern = r"_AUTO_SHAPE_([A-Z_]+)"

    for file_txt in files_txt:
        logging.info(f"Processing file: {file_txt}")
        praises = []
        with open(file_txt, "r", encoding="utf-8") as f:
            text = f.read()

        for praise in tqdm(
            text.split("__END__"), desc="Processing praises", unit="praise"
        ):
            praise_struc = {}
            praise_struc["slides"] = []

            for slide in praise.split("SLIDE_"):
                slide_struc = {}
                slide_num = slide.split("\n")[0].strip()
                if not slide_num:
                    continue

                slide_struc["slide"] = slide_num
                slide_struc["shapes"] = []

                for shape in slide.split("\nSHAPE_"):
                    record_line = False
                    text_inside = ""
                    shapes_struc = {}

                    for line in shape.split("\n"):
                        if "SHAPE_" in line:
                            match = re.search(auto_shape_pattern, line)
                            auto_shape = match.group(1)

                            shapes_struc["auto_shape"] = auto_shape
                            shapes_struc["shape"] = "AUTO_SHAPE"

                        if "HEIGHT_" in line:
                            height = line.split("_")[1]
                            shapes_struc["height"] = int(height)
                        if "TOP_" in line:
                            top = line.split("_")[1]
                            shapes_struc["top"] = int(top)

                        if "END_TEXT" in line:
                            record_line = False

                        if record_line:
                            text_inside += line + "\n"
                        else:
                            if text_inside:
                                shapes_struc["text"] = text_inside.strip()

                        if "START_TEXT" in line:
                            text_inside = ""
                            record_line = True

                    if shapes_struc:
                        slide_struc["shapes"].append(shapes_struc)

                praise_struc["slides"].append(slide_struc)

            # pegar nome/numero do hino antes de salvar
            praises.append(praise_struc)

        new_file = "slides_json\\" + file_txt.split("\\")[1] + ".json"
        with open(new_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(praises, ensure_ascii=False, indent=4))


def main():
    txt2json()


if __name__ == "__main__":
    main()
