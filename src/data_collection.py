"""
data_collection.py
==================
LO1 + LO2 | Manual seksaun 5 (DATA COLLECTION) no seksaun 10 (Kolesaun no
Jestaun Dadus).

Funsaun:
  1. Le'e RAW data husi `data/raw/dataset_turris.json` (snapshot Apify Google
     Maps Reviews Scraper: 1 objetu = 1 lokal turizmu ho lista `reviews`).
  2. "Flatten" ba nivel REVIEW (1 baris = 1 komentariu turista), tanba unidade
     analiza ba sentiment + kategoria mak KADA KOMENTARIU, laos kada lokal.
  3. Junta anotasaun manual kategoria (ground truth) husi
     `data/place_category_labels.csv`.
  4. Grava log kolesaun (`reports/data_collection_log.json`) ho metadata fonte,
     timestamp no kontajen - tuir ezijensia Manual seksaun 10.

STATUS FONTE DADUS (importante ba defeza projetu):
  Grupu SEIDAUK iha API key (Google Places API presiza billing). Tanba ne'e
  sistema uza SNAPSHOT batch, no LA reivindika "real-time". Bainhira API
  disponivel, troka DEIT funsaun `load_raw_data()` iha ne'e ba fetch HTTP +
  polling; modulu seluk hotu (preprocessing, EDA, mining, dashboard) la presiza
  muda tanba sira servisu iha nivel DataFrame. Haree stub `fetch_from_api()`.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    APIFY_ACTOR_ID,
    APIFY_API_BASE,
    APIFY_API_TOKEN,
    APIFY_FETCH_LOG,
    APIFY_INPUT_CONFIG,
    COLLECTION_LOG_JSON,
    DATA_SOURCE,
    PLACE_LABELS_CSV,
    RAW_DATASET,
    RAW_DIR,
    ensure_dirs,
    list_raw_dataset_files,
)

# Kampu husi nivel LOKAL ne'ebe kopia ba kada review (denormalizasaun)
PLACE_FIELDS = {
    "place_id": "placeId",
    "place_name": "title",
    "place_category_google": "categoryName",
    "city": "city",
    "address": "address",
    "place_avg_rating": "totalScore",
    "place_reviews_count": "reviewsCount",
    "scraped_at": "scrapedAt",
    "place_url": "url",
}

REVIEW_FIELDS = {
    "review_id": "reviewId",
    "reviewer_name": "name",
    "review_text": "text",
    "review_stars": "stars",
    "review_date": "publishedAtDate",
    "review_language": "originalLanguage",
    "is_local_guide": "isLocalGuide",
}


def _load_places_json(p: Path) -> list:
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Formatu dataset la tuir ekspetadu (tenke lista objetu lokal).")
    return data


def load_raw_data(path=None) -> list:
    """Le'e raw JSON (lista objetu lokal turizmu ho reviews aninhadu).

    - `path` fo explisitamente  -> le'e DEIT ficheiru ne'e (verifikasaun/demo
      ficheiru ida, haree check_dataset.py).
    - `path=None` (default)     -> le'e HOTU-HOTU `*.json` iha `data/raw/` no
      hamutuk sira (merge by placeId, dedupe review by reviewId - haree
      `merge_places()`). Sistema LA restrik ba naran ficheiru fixu
      `dataset_turris.json`; kualker ficheiru `.json` foun ne'ebe tau iha
      `data/raw/` automatikamente sei prosesa iha kolesaun tuir mai.
    """
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset la iha: {p}")
        return _load_places_json(p)

    files = list_raw_dataset_files()
    if not files:
        raise FileNotFoundError(f"La iha ficheiru .json iha {RAW_DIR}")

    combined: list = []
    used, skipped = [], []
    for f in files:
        try:
            data = _load_places_json(f)
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
            skipped.append((f.name, str(exc)))
            continue
        merge_places(combined, data)
        used.append(f.name)

    if not combined:
        raise ValueError(f"Seidauk iha ficheiru .json validu iha {RAW_DIR}")

    print(f"[load_raw_data] Hamutuk {len(used)} ficheiru raw ({len(combined)} lokal): "
          f"{', '.join(used)}")
    for name, reason in skipped:
        print(f"[load_raw_data] AVIZU: salta '{name}' - {reason[:120]}")
    return combined


def _apify_headers() -> dict:
    return {"Authorization": f"Bearer {APIFY_API_TOKEN}"}


def _apify_actor_url_id() -> str:
    """Apify REST API uza '~' atu separa username/actorName iha URL path
    (hanesan 'compass~crawler-google-places'), LAOS '/' (ne'e sei haree
    hanesan segmentu path extra no fó 404)."""
    return APIFY_ACTOR_ID.replace("/", "~")


def _apify_get(url: str, **params) -> dict:
    resp = requests.get(url, headers=_apify_headers(), params=params, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"Apify API GET {url} falha ({resp.status_code}): {resp.text[:500]}")
    return resp.json()


def _apify_post(url: str, json_body=None, **params) -> dict:
    resp = requests.post(url, headers=_apify_headers(), params=params, json=json_body, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"Apify API POST {url} falha ({resp.status_code}): {resp.text[:500]}")
    return resp.json()


# Pergunta default ba modu SIMPLES (`jumlah_data`) - uza DEIT 1 liña atu
# `maxCrawledPlacesPerSearch` sai IGUAL ba total lokal ne'ebe mai (la multiplika
# ho kantidade liña hanesan modu avansadu/searchStringsArray).
_DEFAULT_SIMPLE_QUERY = "places to visit in Timor-Leste"


def _load_custom_input() -> dict | None:
    """Le'e `apify_request.json` iha project root se ema ona edita. Retorna
    `None` se ficheiru la iha, atu `fetch_from_api()` book nia-an uza INPUT
    husi run successu ikus liu (comportamentu antigu).

    Suporta 2 formatu:
      1. SIMPLES   - {"jumlah_data": N, "cari": "..."(opsional)}
                     N = total lokal ne'ebe mai (1 liña pesquiza deit, atu
                     numeru ne'e LA multiplika), ho detallu KOMPLETU
                     (kontaktu, website, imajen) atu "lengkap".
      2. AVANSADU  - {"searchStringsArray": [...], "maxCrawledPlacesPerSearch": N,
                     "maxReviews": N, ...}  - kampu Apify atual, kontrola
                     kategoria/kota balun hotu-hotu.
    """
    if not APIFY_INPUT_CONFIG.exists():
        return None
    with open(APIFY_INPUT_CONFIG, "r", encoding="utf-8") as f:
        custom = json.load(f)
    if not isinstance(custom, dict):
        raise ValueError(f"{APIFY_INPUT_CONFIG.name} invalido: tenke objetu JSON.")

    if "searchStringsArray" in custom:
        if not custom.get("searchStringsArray"):
            raise ValueError(
                f"{APIFY_INPUT_CONFIG.name} invalido: `searchStringsArray` la bele mamuk."
            )
        return custom

    if "jumlah_data" in custom:
        n = custom["jumlah_data"]
        if not isinstance(n, int) or n < 1:
            raise ValueError(
                f"{APIFY_INPUT_CONFIG.name} invalido: `jumlah_data` tenke numeru >= 1."
            )
        query = custom.get("cari") or _DEFAULT_SIMPLE_QUERY
        return {
            "searchStringsArray": [query],
            "maxCrawledPlacesPerSearch": n,
            "maxReviews": custom.get("maxReviews", 50),
            "language": custom.get("language", "en"),
            "scrapePlaceDetailPage": True,
            "scrapeContacts": True,
            "includeWebResults": True,
            "maxImages": custom.get("maxImages", 10),
        }

    raise ValueError(
        f"{APIFY_INPUT_CONFIG.name} invalido: presiza `jumlah_data` (modu simples) "
        "ka `searchStringsArray` (modu avansadu)."
    )


def _get_last_run_input() -> dict:
    """Hetan INPUT (query/search config) husi run successu ikus liu, atu la
    presiza konfigura fali kota/lokal/bahasa manualmente."""
    url = f"{APIFY_API_BASE}/acts/{_apify_actor_url_id()}/runs/last"
    run = _apify_get(url, status="SUCCEEDED")["data"]
    kv_store_id = run["defaultKeyValueStoreId"]
    input_url = f"{APIFY_API_BASE}/key-value-stores/{kv_store_id}/records/INPUT"
    resp = requests.get(input_url, headers=_apify_headers(), timeout=60)
    if not resp.ok:
        raise RuntimeError(
            f"La bele hetan INPUT husi run successu ikus liu ({resp.status_code}): "
            f"{resp.text[:300]}"
        )
    return resp.json()


def _trigger_run(run_input: dict) -> str:
    url = f"{APIFY_API_BASE}/acts/{_apify_actor_url_id()}/runs"
    data = _apify_post(url, json_body=run_input)["data"]
    return data["id"]


def _wait_for_run(run_id: str, poll_seconds: int = 10, timeout_seconds: int = 1800) -> dict:
    url = f"{APIFY_API_BASE}/actor-runs/{run_id}"
    terminal = {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}
    waited = 0
    while True:
        data = _apify_get(url)["data"]
        status = data["status"]
        if status in terminal:
            if status != "SUCCEEDED":
                raise RuntimeError(
                    f"Apify run {run_id} remata ho status '{status}' "
                    f"({data.get('statusMessage') or 'sem detallu'}). "
                    f"Haree Apify Console -> Runs (posivelmente saldo/usage limit la sufisiente)."
                )
            return data
        time.sleep(poll_seconds)
        waited += poll_seconds
        if waited >= timeout_seconds:
            raise TimeoutError(
                f"Apify run {run_id} liu tempu hein {timeout_seconds}s (status atual: {status})."
            )


def _get_dataset_items(dataset_id: str) -> list:
    """Download dataset items HOTU-HOTU (pagina kompletu, atributu la filtra)."""
    items = []
    limit = 1000
    offset = 0
    url = f"{APIFY_API_BASE}/datasets/{dataset_id}/items"
    while True:
        resp = requests.get(
            url,
            headers=_apify_headers(),
            params={"format": "json", "clean": "true", "limit": limit, "offset": offset},
            timeout=120,
        )
        if not resp.ok:
            raise RuntimeError(f"La bele hetan dataset items ({resp.status_code}): {resp.text[:300]}")
        batch = resp.json()
        if not batch:
            break
        items.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
    return items


def fetch_from_api(poll_seconds: int = 10, timeout_seconds: int = 1800) -> list:
    """
    DATA COLLECTION liu husi Apify REST API (troka ba batch manual download+paste):
      1. Hetan INPUT (kota/lokal/kantidade/bahasa):
         - Se `apify_request.json` iha project root (ema ona edita manualmente)
           -> uza ne'e, atu ema bele kontrola saida ne'ebe request.
         - Se la iha -> book nia-an uza INPUT husi run successu ikus liu
           (comportamentu antigu, la presiza konfigura fali).
      2. Hahu run FOUN ho INPUT ne'e (= "request dadus foun" ho APIFY_API_TOKEN).
      3. Hein to'o run remata (SUCCEEDED/FAILED/... - se saldo/limite Apify la
         sufisiente, erru sei aparese iha-ne'e hanesan iha Apify Console).
      4. Download dataset items KOMPLETU (atributu hotu-hotu, la filtra/flatten -
         preprocessing.py mak atu hili kolona nebee presiza).

    Retorna lista dict (hanesan `load_raw_data()`), atu `flatten_reviews()` no
    pipeline seluk la presiza muda.
    """
    if not APIFY_API_TOKEN:
        raise RuntimeError(
            "APIFY_API_TOKEN la hetan iha .env. Kopia .env.example -> .env no hatama "
            "token lolos (Apify Console -> Settings -> Integrations)."
        )
    run_input = _load_custom_input()
    if run_input is not None:
        print(f"[Apify] Uza INPUT custom husi {APIFY_INPUT_CONFIG.name}")
    else:
        run_input = _get_last_run_input()
        print("[Apify] Uza INPUT husi run successu ikus liu (seidauk iha apify_request.json)")
    run_id = _trigger_run(run_input)
    run_data = _wait_for_run(run_id, poll_seconds=poll_seconds, timeout_seconds=timeout_seconds)
    dataset_id = run_data["defaultDatasetId"]
    items = _get_dataset_items(dataset_id)
    if not isinstance(items, list):
        raise ValueError("Formatu dataset Apify la tuir ekspetadu (tenke lista objetu lokal).")
    return items


def merge_places(existing: list, incoming: list) -> dict:
    """
    Junta lista lokal FOUN (husi fetch_from_api) ba lista lokal EZISTENTE,
    nivel-review (dedupe by reviewId), atu raw la bengka ho lokal duplikadu.
    Duplikata eventual seluk (p.ex. review sem reviewId) sei hasai iha
    preprocessing.preprocess() (dedupe by place_id+review_text+reviewer_name).
    """
    by_id = {p.get("placeId"): p for p in existing if p.get("placeId")}
    stats = {"new_places": 0, "merged_places": 0, "new_reviews": 0}

    for place in incoming:
        pid = place.get("placeId")
        if not pid:
            continue
        if pid in by_id:
            existing_place = by_id[pid]
            existing_review_ids = {
                r.get("reviewId") for r in (existing_place.get("reviews") or []) if r.get("reviewId")
            }
            added = 0
            for rev in (place.get("reviews") or []):
                rid = rev.get("reviewId")
                if rid and rid not in existing_review_ids:
                    existing_place.setdefault("reviews", []).append(rev)
                    existing_review_ids.add(rid)
                    added += 1
            if added:
                stats["merged_places"] += 1
                stats["new_reviews"] += added
            for key in ("scrapedAt", "totalScore", "reviewsCount", "title", "address", "categoryName"):
                if place.get(key) is not None:
                    existing_place[key] = place[key]
            # Haruka lokal ne'ebe foin hetan update ba FINAL/BAIXU liña, atu ema
            # bele hare FASIL entrada ne'ebe foin toka iha fim ficheiru (pedidu
            # user: "masuk di bagian bawah"), laos buka iha mei-mei dadus tuan.
            existing[:] = [p for p in existing if p is not existing_place]
            existing.append(existing_place)
        else:
            existing.append(place)
            by_id[pid] = place
            stats["new_places"] += 1
            stats["new_reviews"] += len(place.get("reviews") or [])

    return stats


def collect_from_api(merge: bool = True, poll_seconds: int = 10, timeout_seconds: int = 1800) -> dict:
    """
    Entry-point ba "request data" liu husi Apify API:
      1. fetch_from_api()           -> dadus foun husi Apify
      2. backup raw ezistente       -> data/raw/dataset_turris_backup_<data>.json
      3. merge (default) ka overwrite ba data/raw/dataset_turris.json
      4. grava log kolesaun

    Hafoin ida-ne'e halao, `py -3.11 src/pipeline.py` (ka dashboard) sei le'e
    raw FOUN automatikamente - preprocessing no pasu seluk LA muda.
    """
    ensure_dirs()
    print(f"[Apify] Hahu request dadus foun husi actor '{APIFY_ACTOR_ID}' ...")
    new_places = fetch_from_api(poll_seconds=poll_seconds, timeout_seconds=timeout_seconds)
    print(f"[Apify] Hetan {len(new_places)} lokal husi run foun.")

    if RAW_DATASET.exists():
        with open(RAW_DATASET, "r", encoding="utf-8") as f:
            existing_places = json.load(f)
        backup_path = RAW_DIR / f"dataset_turris_backup_{datetime.now().strftime('%Y-%m-%d')}.json"
        if not backup_path.exists():
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(existing_places, f, ensure_ascii=False, indent=2)
            print(f"[Apify] Backup raw tuan -> {backup_path}")
    else:
        existing_places = []

    ids_antes = {p.get("placeId") for p in existing_places if p.get("placeId")}

    if merge and existing_places:
        stats = merge_places(existing_places, new_places)
        final_places = existing_places
    else:
        stats = {
            "new_places": len(new_places),
            "merged_places": 0,
            "new_reviews": sum(len(p.get("reviews") or []) for p in new_places),
        }
        final_places = new_places

    with open(RAW_DATASET, "w", encoding="utf-8") as f:
        json.dump(final_places, f, ensure_ascii=False, indent=2)

    stats["total_places"] = len(final_places)
    stats["total_reviews"] = sum(len(p.get("reviews") or []) for p in final_places)
    print(
        f"[Apify] Lokal foun={stats['new_places']}  lokal merge={stats['merged_places']}  "
        f"review foun={stats['new_reviews']}  |  TOTAL lokal={stats['total_places']} "
        f"review={stats['total_reviews']}"
    )
    print(f"[Apify] Grava -> {RAW_DATASET}")

    # Rezultadu nivel-lokal (ho komentariu/review KOMPLETU, laos deit kontajen)
    # atu dashboard bele hatudu "lokal saida mak foin mai" hanesan Apify
    # Console nia Results tab - no dosente bele haree komentariu lolos iha
    # dashboard, laos deit numeru.
    stats["fetched_places"] = [
        {
            "placeId": p.get("placeId"),
            "title": p.get("title"),
            "category": p.get("categoryName"),
            "city": p.get("city"),
            "rating": p.get("totalScore"),
            "reviews_count_google": p.get("reviewsCount"),
            "reviews_fetched": len(p.get("reviews") or []),
            "status": "baru" if p.get("placeId") not in ids_antes else "update",
            "url": p.get("url"),
            "reviews": [
                {
                    "reviewer": r.get("name") or "(anonimu)",
                    "stars": r.get("stars"),
                    "date": r.get("publishedAtDate"),
                    "text": r.get("text") or "",
                }
                for r in (p.get("reviews") or [])
            ],
            # Objetu RAW KOMPLETU (hanesan persis iha data/raw/dataset_turris.json,
            # seidauk limpa/cleaning) - dosente bele haree duplikata, kampu null,
            # metadata Apify atual hotu-hotu, antes pipeline altera buat ida.
            "raw": p,
        }
        for p in new_places
    ]
    stats["run_timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stats["actor_id"] = APIFY_ACTOR_ID
    ensure_dirs()
    with open(APIFY_FETCH_LOG, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats


def flatten_reviews(places: list) -> pd.DataFrame:
    """Konverte lista lokal (ho reviews aninhadu) ba DataFrame nivel-review."""
    rows = []
    for place in places:
        place_meta = {col: place.get(src) for col, src in PLACE_FIELDS.items()}
        location = place.get("location") or {}
        place_meta["lat"] = location.get("lat")
        place_meta["lng"] = location.get("lng")

        reviews = place.get("reviews") or []
        if not reviews:
            # Lokal sem komentariu: mantein iha inventariu (ba mapa/EDA nivel-lokal)
            # maibe la kontribui ba sentiment/kategoria nivel-review.
            row = dict(place_meta)
            row.update({col: None for col in REVIEW_FIELDS})
            row["has_review"] = False
            rows.append(row)
            continue

        for r in reviews:
            row = dict(place_meta)
            row.update({col: r.get(src) for col, src in REVIEW_FIELDS.items()})
            row["has_review"] = True
            rows.append(row)

    return pd.DataFrame(rows)


def attach_place_labels(df: pd.DataFrame, labels_path=None) -> pd.DataFrame:
    """
    Junta anotasaun manual kategoria (ground truth nivel-lokal) ba DataFrame
    nivel-review. Kolona resultadu:
        category_label      -> kategoria lolos (Historical/Religious/Traditional/Modern)
        label_confidence    -> high / medium / low
        label_evidence      -> justifikasaun anotasaun
    """
    labels_path = Path(labels_path) if labels_path else PLACE_LABELS_CSV
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Ficheiru anotasaun la iha: {labels_path}\n"
            "Halao uluk: py -3.11 src/annotate_places.py"
        )
    labels = pd.read_csv(labels_path, encoding="utf-8-sig")
    keep = ["place_id", "category_label", "label_confidence", "label_evidence"]
    return df.merge(labels[keep], on="place_id", how="left")


def write_collection_log(df: pd.DataFrame, dataset_path=None) -> dict:
    """Grava metadata kolesaun (Manual seksaun 10). `dataset_path=None`
    (default) significa modu merge-hotu -> grava LISTA ficheiru `data/raw/*.json`
    ne'ebe efetivamente kontribui ba `df`."""
    ensure_dirs()
    if dataset_path is None:
        files = list_raw_dataset_files()
        dataset_files = [f.name for f in files]
        dataset_size_bytes = sum(f.stat().st_size for f in files)
    else:
        dataset_path = Path(dataset_path)
        dataset_files = [dataset_path.name]
        dataset_size_bytes = dataset_path.stat().st_size
    scraped = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True, format="mixed")
    log = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_files": dataset_files,
        "dataset_size_bytes": dataset_size_bytes,
        "data_source": DATA_SOURCE,
        "counts": {
            "places": int(df["place_id"].nunique()),
            "rows": int(len(df)),
            "rows_with_review": int(df["has_review"].sum()),
            "places_without_review": int((~df["has_review"]).sum()),
            "reviews_with_text": int(df["review_text"].notna().sum()),
        },
        "collection_window": {
            "scraped_at_min": str(scraped.min()) if scraped.notna().any() else None,
            "scraped_at_max": str(scraped.max()) if scraped.notna().any() else None,
        },
    }
    with open(COLLECTION_LOG_JSON, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    return log


def collect(path=None, with_labels: bool = True, write_log: bool = True) -> pd.DataFrame:
    """Entry-point ba pipeline: load -> flatten -> junta label -> log.

    `path=None` (default) -> le'e no hamutuk TODO `*.json` iha `data/raw/`
    (haree `load_raw_data()`). Fo `path` explisitu deit se presiza verifika
    ficheiru ida espesifiku."""
    raw = load_raw_data(path)
    df = flatten_reviews(raw)
    if with_labels:
        df = attach_place_labels(df)
    if write_log:
        write_collection_log(df, path)
    return df


if __name__ == "__main__":
    if "--fetch-api" in sys.argv:
        collect_from_api()
        sys.exit(0)

    df = collect()
    print(f"Total lokal        : {df['place_id'].nunique()}")
    print(f"Total baris        : {len(df)}")
    print(f"Baris ho review    : {int(df['has_review'].sum())}")
    print(f"Review ho teksu    : {int(df['review_text'].notna().sum())}")
    print(f"\nDistribuisaun kategoria (ground truth, nivel-review):")
    print(df["category_label"].value_counts().to_string())
    print(f"\nLog kolesaun -> {COLLECTION_LOG_JSON}")
