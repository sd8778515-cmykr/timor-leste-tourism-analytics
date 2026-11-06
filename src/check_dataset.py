"""
check_dataset.py
================
FERRAMENTA VERIFIKASAUN ba dataset foun husi Apify.

Uza ida-ne'e ANTES halao pipeline, bainhira ita troka
`data/raw/dataset_turris.json` ho export foun husi Apify. Nia verifika:

  1. Formatu JSON valido no estrutura ne'ebe pipeline espera
  2. Kampu obrigatoriu iha ka lae (placeId, title, reviews, location, ...)
  3. Kuantidade lokal, review, review ho teksu
  4. LOKAL FOUN ne'ebe seidauk iha anotasaun manual kategoria
     (importante! senaun sira sei simu label default "Modern")

Halao:
    py -3.11 src\\check_dataset.py                    # verifika ficheiru atual
    py -3.11 src\\check_dataset.py C:\\path\\foun.json  # verifika ficheiru seluk
"""

import json
import sys
from collections import Counter
from pathlib import Path

# Windows PowerShell lama bele uza code page ne'ebe la suporta karakter Tetun
# no naran lokal. Hatudu output UTF-8 atu verifikasaun demo la falla bainhira
# imprime karakter hanesan ã, ó ka aksara non-Latin.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PLACE_LABELS_CSV, list_raw_dataset_files

# Kampu ne'ebe pipeline uza (nivel lokal)
REQUIRED_PLACE = ["placeId", "title"]
OPTIONAL_PLACE = ["categoryName", "city", "address", "totalScore",
                  "reviewsCount", "scrapedAt", "url", "location", "reviews"]
# Kampu ne'ebe pipeline uza (nivel review)
REVIEW_FIELDS = ["reviewId", "name", "text", "stars", "publishedAtDate",
                 "originalLanguage", "isLocalGuide"]

OK, WARN, BAD = "[OK]  ", "[AVIZU]", "[SALA]"


def check(path: Path) -> bool:
    print("=" * 72)
    print(f"  VERIFIKASAUN DATASET: {path.name}")
    print("=" * 72)

    if not path.exists():
        print(f"{BAD} Ficheiru la iha: {path}")
        return False
    print(f"{OK} Ficheiru iha ({path.stat().st_size / 1024:.0f} KB)")

    # --- 1. JSON valido? ---------------------------------------------------
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"{BAD} JSON la valido: {exc}")
        print("      -> Iha Apify, export ho formatu **JSON** (laos JSONL/CSV).")
        return False
    print(f"{OK} JSON valido")

    # --- 2. Estrutura ------------------------------------------------------
    if not isinstance(data, list):
        print(f"{BAD} Formatu tenke LISTA objetu. Hetan: {type(data).__name__}")
        return False
    if not data:
        print(f"{BAD} Dataset mamuk (0 item).")
        return False
    print(f"{OK} Lista ho {len(data)} item")

    first = data[0]
    if not isinstance(first, dict):
        print(f"{BAD} Item la'os objetu JSON.")
        return False

    # Deteta se item = LOKAL (ho reviews aninhadu) ka REVIEW (flat)
    has_reviews_key = any(isinstance(d, dict) and "reviews" in d for d in data[:20])
    looks_like_review = "reviewId" in first or ("text" in first and "stars" in first
                                                and "title" not in first)
    if looks_like_review and not has_reviews_key:
        print(f"{BAD} Item sira hanesan REVIEW individual (formatu flat).")
        print("      Pipeline espera 1 item = 1 LOKAL ho lista `reviews` aninhadu.")
        print("      -> Iha Apify actor `compass/crawler-google-places`, iha")
        print("         konfigurasaun, HABILITA opsaun scrape reviews no export")
        print("         dataset PRINSIPAL (laos dataset review separadu).")
        return False
    if not has_reviews_key:
        print(f"{WARN} La hetan kampu `reviews` iha item 20 dahuluk.")
        print("      Sistema sei le'e, maibe sentiment/clustering la iha dadus.")

    # --- 3. Kampu obrigatoriu ---------------------------------------------
    missing_required = Counter()
    for d in data:
        for k in REQUIRED_PLACE:
            if not d.get(k):
                missing_required[k] += 1
    if missing_required:
        for k, n in missing_required.items():
            lvl = BAD if k == "placeId" else WARN
            print(f"{lvl} Kampu `{k}` mamuk iha {n}/{len(data)} lokal")
        if missing_required.get("placeId"):
            print("      -> `placeId` obrigatoriu (xave ba junta anotasaun kategoria).")
            return False
    else:
        print(f"{OK} Kampu obrigatoriu ({', '.join(REQUIRED_PLACE)}) kompletu")

    for k in OPTIONAL_PLACE:
        n = sum(1 for d in data if d.get(k) in (None, "", [], {}))
        if n:
            print(f"{WARN} Kampu opsional `{k}` mamuk iha {n}/{len(data)} lokal")

    # --- 4. Kontajen review ------------------------------------------------
    n_reviews = n_text = n_stars = 0
    review_field_missing = Counter()
    for d in data:
        for r in (d.get("reviews") or []):
            n_reviews += 1
            if isinstance(r.get("text"), str) and r["text"].strip():
                n_text += 1
            if r.get("stars") is not None:
                n_stars += 1
            for k in REVIEW_FIELDS:
                if r.get(k) is None:
                    review_field_missing[k] += 1

    print()
    print("  KONTAJEN")
    print(f"     Lokal turizmu       : {len(data)}")
    print(f"     Total review        : {n_reviews}")
    print(f"     Review ho teksu     : {n_text}"
          + (f" ({n_text / n_reviews * 100:.1f}%)" if n_reviews else ""))
    print(f"     Review ho rating    : {n_stars}")
    print(f"     Lokal sem review    : {sum(1 for d in data if not (d.get('reviews') or []))}")

    if n_text < 30:
        print(f"{WARN} Review ho teksu < 30 -> clustering no model supervisadu")
        print("      bele la funsiona di'ak (presiza dadus teksu sufisiente).")

    # --- 5. Lokal foun ne'ebe seidauk iha anotasaun kategoria -------------
    print()
    print("  ANOTASAUN KATEGORIA (ground truth)")
    try:
        import pandas as pd
        from annotate_places import EXPERT_LABELS, _normalize

        known = {_normalize(k) for k in EXPERT_LABELS}
        new_places = [d.get("title") for d in data
                      if _normalize(d.get("title")) not in known]

        if not new_places:
            print(f"{OK} Lokal hotu-hotu ({len(data)}) iha anotasaun manual ona")
        else:
            print(f"{WARN} {len(new_places)} lokal FOUN seidauk iha anotasaun manual:")
            for name in new_places[:25]:
                print(f"         - {name}")
            if len(new_places) > 25:
                print(f"         ... no {len(new_places) - 25} tan")
            print()
            print("      Lokal sira-ne'e sei simu label default `Modern`")
            print("      (confidence = `low`). Atu hadia:")
            print(f"         1. py -3.11 src\\annotate_places.py")
            print(f"         2. Loke {PLACE_LABELS_CSV.name} iha Excel")
            print("         3. Troka kolona `category_label` ba lokal foun sira")
            print("            (Historical / Religious / Traditional / Modern)")
            print("         4. Salva hanesan CSV, depois halao pipeline")
    except Exception as exc:  # pragma: no cover
        print(f"{WARN} La bele verifika anotasaun: {exc}")

    print()
    print("=" * 72)
    print("  DATASET KOMPATIVEL — bele halao: py -3.11 src\\pipeline.py")
    print("=" * 72)
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        targets = [Path(sys.argv[1])]
    else:
        targets = list_raw_dataset_files()
        if not targets:
            print(f"{BAD} La iha ficheiru .json iha data/raw/")
            sys.exit(1)
        print(f"La iha naran ficheiru fixu - verifika HOTU-HOTU {len(targets)} "
              f"ficheiru .json iha data/raw/ (sistema prosesa hotu-hotu, laos deit ida):\n")

    results = [check(t) for t in targets]
    if len(targets) > 1:
        print()
        print("=" * 72)
        print(f"  REZUMU: {sum(results)}/{len(targets)} ficheiru pasa verifikasaun")
        print("=" * 72)
    sys.exit(0 if all(results) else 1)
