from pptx2txt import pptx2txt
from txt2json import txt2json
from json2sql import json2sql


def main():
    pptx2txt()
    txt2json()
    json2sql(3)


if __name__ == "__main__":
    main()
