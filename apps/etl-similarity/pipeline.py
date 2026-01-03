from pathlib import Path
import argparse
import logging
from datetime import datetime

from extract_data import extract
import similarities
import processes
import config
from utils import setup_logger, PipelineTracker


def pipeline(
    assets_folder: Path, coletanea_id: int | None = None, log_level: int = logging.INFO
) -> None:
    """Run the full ETL pipeline sequentially with logging and tracking."""

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = setup_logger(assets_folder, run_id, level=log_level)
    tracker = PipelineTracker(assets_folder, run_id=run_id)

    logger.info(f"Pipeline started (run_id={run_id}).")
    logger.info(f"Assets folder: {assets_folder}")

    # 1) extract
    step = "extract"
    tracker.start(step)
    logger.info("[extract] Loading hymns from database...")
    try:
        hinos_extract = extract(coletanea_id)
        tracker.end(
            step,
            success=True,
            extra={"rows": len(hinos_extract), "columns": list(hinos_extract.columns)},
        )
        logger.info(f"[extract] Loaded {len(hinos_extract)} rows.")
    except Exception as e:
        tracker.fail(step, str(e))
        logger.exception("[extract] Failed.")
        raise

    # 2) similarity_title
    step = "similarity_title"
    tracker.start(step)
    logger.info("[similarity_title] Calculating title similarities...")
    try:
        similarities.similarity_title(hinos_extract, assets_folder)
        artifact = assets_folder / "similarity_matrix_titles.pkl"
        tracker.end(step, success=True, extra={"artifact": str(artifact)})
        logger.info("[similarity_title] Title similarity saved.")
    except Exception as e:
        tracker.fail(step, str(e))
        logger.exception("[similarity_title] Failed.")
        raise

    # 3) process_tokens
    step = "process_tokens"
    tracker.start(step)
    logger.info("[process_tokens] Processing tokens...")
    try:
        stopwords_path = config.SHARED_DIR / "assets"
        hinos_tokens = processes.process_tokens(hinos_extract, stopwords_path)
        tracker.end(step, success=True, extra={"rows": len(hinos_tokens)})
        logger.info("[process_tokens] Tokens processed.")
    except Exception as e:
        tracker.fail(step, str(e))
        logger.exception("[process_tokens] Failed.")
        raise

    # 4) process_ngrams + similarity_word
    step = "process_ngrams"
    tracker.start(step)
    logger.info("[process_ngrams] Processing n-grams...")
    try:
        hinos_ngrams = processes.process_ngrams(hinos_tokens)
        tracker.end(step, success=True, extra={"rows": len(hinos_ngrams)})
        logger.info("[process_ngrams] N-grams processed.")
    except Exception as e:
        tracker.fail(step, str(e))
        logger.exception("[process_ngrams] Failed.")
        raise

    step = "similarity_word"
    tracker.start(step)
    logger.info("[similarity_word] Calculating word TF-IDF similarities...")
    try:
        similarities.similarity_word(hinos_ngrams, assets_folder)
        artifact = assets_folder / "similarity_matrix_words.pkl"
        tracker.end(step, success=True, extra={"artifact": str(artifact)})
        logger.info("[similarity_word] Word TF-IDF similarity saved.")
    except Exception as e:
        tracker.fail(step, str(e))
        logger.exception("[similarity_word] Failed.")
        raise

    # 5) process_word_embeddings + similarity_word_embeddings
    step = "process_word_embeddings"
    tracker.start(step)
    logger.info("[process_word_embeddings] Computing word embeddings...")
    try:
        model_path = config.SHARED_DIR / "models" / config.FASTTEXT_MODEL_NAME
        hinos_word_emb = processes.process_word_embeddings(hinos_ngrams, model_path)
        tracker.end(step, success=True, extra={"rows": len(hinos_word_emb)})
        logger.info("[process_word_embeddings] Word embeddings computed.")
    except Exception as e:
        tracker.fail(step, str(e))
        logger.exception("[process_word_embeddings] Failed.")
        raise

    step = "similarity_word_embeddings"
    tracker.start(step)
    logger.info(
        "[similarity_word_embeddings] Calculating word embedding similarities..."
    )
    try:
        similarities.similarity_word_embeddings(hinos_word_emb, assets_folder)
        artifact = assets_folder / "similarity_matrix_word_embeddings.pkl"
        tracker.end(step, success=True, extra={"artifact": str(artifact)})
        logger.info("[similarity_word_embeddings] Word embedding similarity saved.")
    except Exception as e:
        tracker.fail(step, str(e))
        logger.exception("[similarity_word_embeddings] Failed.")
        raise

    # 6) process_sentence_embeddings + similarity_sentence_embeddings
    step = "process_sentence_embeddings"
    tracker.start(step)
    logger.info("[process_sentence_embeddings] Computing sentence embeddings...")
    try:
        hinos_sent_emb = processes.process_sentence_embeddings(
            hinos_word_emb, config.SENTENCE_TRANSFORMER_MODEL
        )
        tracker.end(step, success=True, extra={"rows": len(hinos_sent_emb)})
        logger.info("[process_sentence_embeddings] Sentence embeddings computed.")
    except Exception as e:
        tracker.fail(step, str(e))
        logger.exception("[process_sentence_embeddings] Failed.")
        raise

    step = "similarity_sentence_embeddings"
    tracker.start(step)
    logger.info(
        "[similarity_sentence_embeddings] Calculating sentence embedding similarities..."
    )
    try:
        similarities.similarity_sentence_embeddings(hinos_sent_emb, assets_folder)
        artifact = assets_folder / "similarity_matrix_sentence_embeddings.pkl"
        tracker.end(step, success=True, extra={"artifact": str(artifact)})
        logger.info(
            "[similarity_sentence_embeddings] Sentence embedding similarity saved."
        )
    except Exception as e:
        tracker.fail(step, str(e))
        logger.exception("[similarity_sentence_embeddings] Failed.")
        raise

    # 7) process_emotions + similarity_emotions
    step = "process_emotions"
    tracker.start(step)
    logger.info("[process_emotions] Processing emotions...")
    try:
        hinos_emotions = processes.process_emotions(
            hinos_sent_emb, config.EMOTION_MODEL
        )
        tracker.end(step, success=True, extra={"rows": len(hinos_emotions)})
        logger.info("[process_emotions] Emotions processed.")
    except Exception as e:
        tracker.fail(step, str(e))
        logger.exception("[process_emotions] Failed.")
        raise

    step = "similarity_emotions"
    tracker.start(step)
    logger.info("[similarity_emotions] Calculating emotion similarities...")
    try:
        similarities.similarity_emotions(hinos_emotions, assets_folder)
        artifact = assets_folder / "similarity_matrix_emotions.pkl"
        tracker.end(step, success=True, extra={"artifact": str(artifact)})
        logger.info("[similarity_emotions] Emotion similarity saved.")
    except Exception as e:
        tracker.fail(step, str(e))
        logger.exception("[similarity_emotions] Failed.")
        raise

    # 8) final_pickle
    step = "final_pickle"
    tracker.start(step)
    logger.info("[final_pickle] Saving final dataframe with all features...")
    try:
        out_path = assets_folder / "hinos_analise_final.pkl"
        hinos_emotions.to_pickle(out_path)
        tracker.end(step, success=True, extra={"artifact": str(out_path)})
        logger.info("[final_pickle] Final dataframe saved.")
    except Exception as e:
        tracker.fail(step, str(e))
        logger.exception("[final_pickle] Failed.")
        raise

    logger.info("Pipeline finished.")


def main() -> None:
    parser = argparse.ArgumentParser(description="ETL Pipeline for Hymn Similarities")
    parser.add_argument(
        "--assets-folder",
        type=Path,
        default=config.ASSETS_DIR,
        help="Path to the assets folder",
    )
    parser.add_argument(
        "--coletanea-id",
        type=int,
        default=None,
        help="Optional coletanea ID to filter hymns.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console and file logging level.",
    )

    args = parser.parse_args()
    assets_folder: Path = Path(args.assets_folder)
    assets_folder.mkdir(parents=True, exist_ok=True)

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    log_level = level_map.get(args.log_level, logging.INFO)

    coletanea_id = args.coletanea_id

    pipeline(assets_folder, coletanea_id=coletanea_id, log_level=log_level)


if __name__ == "__main__":
    main()
