"""
sentiment_analysis.py
=====================
LO4 | Manual seksaun 13 - Sentiment Analysis (Positive / Neutral / Negative).

MODEL 1 (baseline): LEXICON-BASED. Manual seksaun 13 lista esplisitamente
"lexicon approach" hanesan metodu validu ba Sentiment. Hili tanba dataset LA
IHA label sentiment manual, no lexicon transparente (bele audita liafuan ida-idak).

Lojika:
  1. Review ho TEKSU  -> score = (n_liafuan_positivu - n_liafuan_negativu),
     ho tratamentu NEGASAUN ('la diak', 'tidak bagus', 'not good' -> negativu).
        score > 0 -> Positive | score < 0 -> Negative | score == 0 -> Neutral
  2. Review SEM teksu (rating-only) -> fallback husi `review_stars`:
        >= 4 -> Positive | == 3 -> Neutral | <= 2 -> Negative
     Marka iha kolona `sentiment_source` = 'rating'.
  3. Sem teksu NO sem rating -> 'Unknown' (esklui husi metrika, hatudu iha
     dashboard hanesan transparensia).

GROUND TRUTH ba evaluation (Manual seksaun 14):
  Dataset la iha label manual, tanba ne'e uza `sentiment_true_proxy` =
  label ne'ebe deriva husi RATING (1-5 estrela) hanesan PROXY. Ida-ne'e
  metodu standar bainhira la iha anotasaun; nia limitasaun (rating no teksu
  bele la konkorda - ex: 5 estrela ho komentariu neutru) dokumenta iha
  README seksaun 10 no iha relatoriu Chapter 6.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SENTIMENTS

# --------------------------------------------------------------- LEXICON ----
# Multi-lingua: Tetun (tet), Indonesia (id), Portugues (pt), Ingles (en).
POSITIVE_WORDS = set("""
bagus indah cantik bersih ramah nyaman recommended rekomendasi mantap keren
asik asyik seru senang suka puas menakjubkan terbaik spektakuler worth murah
gratis segar nikmat sejuk aman terawat jernih dingin damai memuaskan
diak furak kmanek kapas midar loos klean seguru haksolok kontente bonitu
bonito lindo lindissima linda maravilhoso maravilha excelente bom boa optimo
otimo agradavel tranquilo paraiso espetacular fantastico incrivel recomendo
beautiful amazing great nice love good clean friendly best wonderful enjoy
fun recommend excellent stunning peaceful relax relaxing refreshing lovely
impressive spectacular breathtaking perfect awesome pretty gorgeous scenic
""".split())

NEGATIVE_WORDS = set("""
buruk jelek kotor kecewa mahal rusak sepi mengecewakan bau sampah susah sulit
jauh licin bahaya berbahaya kurang menyeramkan hancur terbengkalai
aat lakon nulan problema susar perigu manas
sujo feio mau caro perigoso longe dificil abandonado degradado estragado
pessimo horrivel triste sujeira
bad dirty disappointed disappointing expensive damaged broken unsafe poor
worst terrible awful avoid dangerous difficult unfriendly smelly trash
garbage overpriced dilapidated stinky steep slippery neglected ruined crowded
""".split())

# Liafuan ne'ebe fila signifikadu husi liafuan tuir mai
NEGATORS = {"tidak", "tak", "la", "nao", "not", "no", "sem", "kurang", "belum",
            "never", "without", "dont", "doesnt", "isnt", "wasnt"}

# Fronteira rating -> label (dokumentadu, bele ajusta husi dosente)
RATING_POSITIVE_MIN = 4
RATING_NEUTRAL = 3


def sentiment_from_text(text_clean: str) -> tuple:
    """
    Lexicon scoring ho tratamentu negasaun.
    Retorna (label, score, n_pos, n_neg). Se teksu mamuk -> (None, 0, 0, 0).
    """
    if not text_clean:
        return None, 0, 0, 0

    tokens = text_clean.split()
    n_pos = n_neg = 0
    for i, tok in enumerate(tokens):
        negated = i > 0 and tokens[i - 1] in NEGATORS
        if tok in POSITIVE_WORDS:
            if negated:
                n_neg += 1
            else:
                n_pos += 1
        elif tok in NEGATIVE_WORDS:
            if negated:
                n_pos += 1
            else:
                n_neg += 1

    score = n_pos - n_neg
    if score > 0:
        label = "Positive"
    elif score < 0:
        label = "Negative"
    else:
        label = "Neutral"
    return label, score, n_pos, n_neg


def sentiment_from_rating(stars) -> str | None:
    """Konverte rating 1-5 ba label sentiment (fallback no ground-truth proxy)."""
    if pd.isna(stars):
        return None
    stars = float(stars)
    if stars >= RATING_POSITIVE_MIN:
        return "Positive"
    if stars == RATING_NEUTRAL:
        return "Neutral"
    return "Negative"


def apply_sentiment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplika Model 1 (lexicon) + fallback rating. Hatama kolona:
        sentiment_lexicon    -> predisaun husi teksu (None se sem teksu)
        sentiment_score      -> n_pos - n_neg
        sentiment_n_pos/neg  -> kuantidade liafuan lexicon ne'ebe match
        sentiment_true_proxy -> ground truth proxy husi rating
        sentiment            -> label FINAL (teksu > rating > Unknown)
        sentiment_source     -> 'text' | 'rating' | 'none'
    """
    df = df.copy()

    res = df["text_clean"].apply(sentiment_from_text)
    df["sentiment_lexicon"] = res.apply(lambda r: r[0])
    df["sentiment_score"] = res.apply(lambda r: r[1])
    df["sentiment_n_pos"] = res.apply(lambda r: r[2])
    df["sentiment_n_neg"] = res.apply(lambda r: r[3])

    df["sentiment_true_proxy"] = df["review_stars"].apply(sentiment_from_rating)

    def _final(row):
        # NOTA teknika: bainhira None grava iha DataFrame, pandas troka ba NaN,
        # no `NaN is not None` sai True (sala!). Tenke uza pd.notna().
        if row["has_text"] and pd.notna(row["sentiment_lexicon"]):
            return row["sentiment_lexicon"], "text"
        if pd.notna(row["sentiment_true_proxy"]):
            return row["sentiment_true_proxy"], "rating"
        return "Unknown", "none"

    final = df.apply(_final, axis=1)
    df["sentiment"] = final.apply(lambda r: r[0])
    df["sentiment_source"] = final.apply(lambda r: r[1])
    return df


def sentiment_summary(df: pd.DataFrame) -> dict:
    """Rezumu badak ba EDA/log."""
    return {
        "distribution": df["sentiment"].value_counts().to_dict(),
        "source": df["sentiment_source"].value_counts().to_dict(),
        "labels_used": SENTIMENTS + ["Unknown"],
        "lexicon_size": {"positive": len(POSITIVE_WORDS), "negative": len(NEGATIVE_WORDS)},
    }


if __name__ == "__main__":
    from data_collection import collect
    from preprocessing import preprocess

    raw = collect()
    clean, _ = preprocess(raw)
    out = apply_sentiment_analysis(clean)

    print("Distribuisaun sentiment FINAL:")
    print(out["sentiment"].value_counts().to_string())
    print("\nFonte label:")
    print(out["sentiment_source"].value_counts().to_string())
    print("\nExemplu:")
    print(out.loc[out["has_text"], ["place_name", "review_stars", "sentiment",
                                    "sentiment_score", "review_text"]]
          .head(10).to_string(max_colwidth=60))
