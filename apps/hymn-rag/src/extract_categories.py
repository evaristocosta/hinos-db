import unicodedata


# ===== EXTRAÇÃO DETERMINÍSTICA DE FILTROS =====
def _normalize_for_matching(text: str) -> str:
    """Normaliza texto para matching: remove acentos, converte a minúsculas, remove espaços extras."""
    if not text:
        return ""
    # Remove acentos
    nfkd = unicodedata.normalize("NFKD", text)
    text_no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Minúsculas e espaço único
    return " ".join(text_no_accents.lower().split())


def _find_matches_in_text(query: str, candidates: list) -> list:
    """
    Busca candidatos dentro da query usando estratégia híbrida:
    1. Substring match exato (normalizado) - score 1.0
    2. Partial token match - score baseado em tokens comuns
    3. Fuzzy matching (thefuzz) como fallback

    Retorna lista de (candidate, score) ordenada por score.
    """
    from thefuzz import fuzz

    matches = []
    query_norm = _normalize_for_matching(query)
    query_tokens = set(query_norm.split())

    for candidate in candidates:
        candidate_norm = _normalize_for_matching(candidate)

        # Estratégia 1: Substring match direto (melhor caso)
        if candidate_norm in query_norm:
            matches.append((candidate, 1.0))
            continue

        # Estratégia 2: Match por tokens (bom para variações de ordem)
        candidate_tokens = set(candidate_norm.split())
        if len(candidate_tokens) == 0:
            continue

        # Calcula overlap de tokens
        common_tokens = query_tokens & candidate_tokens
        token_ratio = len(common_tokens) / len(candidate_tokens)

        # Se a maioria dos tokens da categoria está na query, considera match
        if token_ratio >= 0.7:
            matches.append((candidate, token_ratio))
            continue

        # Estratégia 3: Fuzzy matching como fallback (para typos)
        # Usa partial_ratio que é ideal para encontrar substring fuzzy
        if len(candidate_norm.split()) <= 4:
            ratio = fuzz.partial_ratio(candidate_norm, query_norm) / 100.0
            # Threshold mais alto para fuzzy, pois já falhou nos outros métodos
            if ratio >= 0.6:
                matches.append((candidate, ratio * 0.8))  # Penaliza fuzzy match

    # Ordena por score (descendente)
    return sorted(matches, key=lambda x: x[1], reverse=True)


def extract_filters_deterministic(
    question: str, categorias_dict: dict, coletaneas_dict: dict
) -> dict:
    """
    Extrai filtros de forma determinística sem usar LLM.

    Estratégia:
    1. Busca categorias/coletâneas contidas na query (substring + token matching)
    2. Remove as referências encontradas da query
    3. Retorna filtros e query limpa
    """
    question_lower = question.lower()
    found_categorias = []
    found_coletaneas = []

    # Extrai possíveis categorias
    categoria_names = list(categorias_dict.keys())
    cat_matches = _find_matches_in_text(question_lower, categoria_names)

    # Aceita matches com score >= 0.7 (alta confiança)
    for cat_name, score in cat_matches:
        if score >= 0.7:
            found_categorias.append(cat_name)
            # Remove a referência da pergunta (para limpeza)
            question_lower = question_lower.replace(cat_name.lower(), " ")

    # Extrai possíveis coletâneas
    coletanea_names = list(coletaneas_dict.keys())
    col_matches = _find_matches_in_text(question_lower, coletanea_names)

    for col_name, score in col_matches:
        if score >= 0.7:
            found_coletaneas.append(col_name)
            # Remove a referência da pergunta
            question_lower = question_lower.replace(col_name.lower(), " ")

    # Limpa a query removendo espaços extras
    cleaned_query = " ".join(question_lower.split())

    return {
        "categorias": found_categorias if found_categorias else None,
        "coletaneas": found_coletaneas if found_coletaneas else None,
        "search_query": cleaned_query if cleaned_query else question,
        "matches_info": {
            "categorias_scores": cat_matches[:3],
            "coletaneas_scores": col_matches[:3],
        },
    }
