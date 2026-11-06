"""  
supervised_models.py
====================
LO4 + LO5 | Manual seksaun 13 (Classification: Decision Tree, Random Forest,
SVM, Naive Bayes) no seksaun 14 (Metrics: Accuracy, Precision, Recall,
F1-score, Confusion Matrix).

MODEL 2 (statistical / machine learning), atu KOMPARA ho MODEL 1 rule-based:
    - Multinomial Naive Bayes   (NB)   -> baseline probabilistiku ba teksu
    - Logistic Regression       (LR)   -> linear, ho probabilidade
    - Linear SVM                (SVM)  -> forte ba TF-IDF dimensaun aas
    - Random Forest             (RF)   -> non-linear, ensemble

Manual seksaun 9 (plano 16 semana) ezije: semana 10 "Implementa model 1",
semana 11 "Implementa/kompara model 2". Modulu ne'e mak realiza pasu ne'e.

DUA TAREFA klasifikasaun:
  A. SENTIMENT  -> target = `sentiment_true_proxy` (label deriva husi rating)
  B. KATEGORIA  -> target = `category_label` (anotasaun manual, ground truth)

Metodolojia (atu evita resultadu enganadór):
  * TF-IDF FIT DEIT iha dadus TRAIN, depois transform ba test (la iha leakage).
  * Split mai husi kolona `split` ne'ebe preprocessing.py kria ho stratifikasaun.
  * Aumenta ho Stratified K-Fold Cross-Validation ba estimativa liu estavel
    (dataset ki'ik -> resultadu split ida deit bele varia barak).
  * Metrika macro-average uza tanba klase LA BALANSU (Manual seksaun 14).
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CATEGORIES,
    CATEGORY_MODEL_PATH,
    CV_FOLDS,
    MODEL_DIR,
    RANDOM_STATE,
    SENTIMENTS,
    SENTIMENT_MODEL_PATH,
)

warnings.filterwarnings("ignore", category=UserWarning)

TEXT_COL = "tokens_joined"


def _make_models(random_state: int = RANDOM_STATE) -> dict:
    """Definisaun model sira. Pipeline inklui TF-IDF atu CV la iha leakage."""
    def tfidf():
        return TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)

    # ======================================================================
    # DEMO DOSEN 1 — EMPAT MODEL YANG DIBANDINGKAN
    # Setiap model menerima teks yang diubah menjadi angka oleh TF-IDF.
    # Model belum dipilih di sini; semuanya akan dilatih dan dievaluasi.
    # ======================================================================
    return {
        # DEMO DOSEN 1A — NAIVE BAYES
        # Model probabilistik yang umum digunakan sebagai baseline klasifikasi teks.
        "Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB(alpha=0.3))]),

        # DEMO DOSEN 1B — LOGISTIC REGRESSION
        # Model linear dengan class_weight balanced untuk kelas tidak seimbang.
        "Logistic Regression": Pipeline([
            ("tfidf", tfidf()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=random_state)),
        ]),

        # DEMO DOSEN 1C — LINEAR SVM
        # Model linear yang kuat untuk fitur teks TF-IDF berdimensi tinggi.
        "Linear SVM": Pipeline([
            ("tfidf", tfidf()),
            ("clf", LinearSVC(class_weight="balanced", random_state=random_state)),
        ]),

        # DEMO DOSEN 1D — RANDOM FOREST
        # Model ensemble yang menggabungkan banyak decision tree.
        "Random Forest": Pipeline([
            ("tfidf", tfidf()),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                           random_state=random_state, n_jobs=-1)),
        ]),
    }


def _metric_block(y_true, y_pred, labels: list) -> dict:
    """
    Kalkula konjuntu metrika kompletu tuir Manual seksaun 14.

    Inklui DETEKSAUN MODELU DEJENERADU: modelu ida ne'ebe preve DEIT klase ida
    (ex: hotu "Positive") bele hetan accuracy aas iha dataset la balansu, maibe
    la iha valor prátiku. Ida-ne'e marka iha kampu `is_degenerate` no `warning`,
    no hatudu iha dashboard/relatoriu — importante ba diskusaun Chapter 6.
    """
    present = list(labels)
    report = classification_report(
        y_true, y_pred, labels=present, output_dict=True, zero_division=0
    )
    n_pred_classes = len(set(y_pred))
    degenerate = n_pred_classes <= 1
    warning = None
    if degenerate:
        only = list(set(y_pred))[0] if n_pred_classes == 1 else "N/A"
        warning = (
            f"Degenerate model: every test record was predicted as '{only}'. "
            "The high accuracy is caused by class imbalance rather than reliable "
            "classification. Use macro F1 and cross-validation for interpretation."
        )
    return {
        "n_predicted_classes": int(n_pred_classes),
        "is_degenerate": bool(degenerate),
        "warning": warning,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(
            y_true, y_pred, labels=present, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(
            y_true, y_pred, labels=present, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(
            y_true, y_pred, labels=present, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(
            y_true, y_pred, labels=present, average="weighted", zero_division=0)), 4),
        "per_class": {
            lab: {
                "precision": round(float(report[lab]["precision"]), 4),
                "recall": round(float(report[lab]["recall"]), 4),
                "f1_score": round(float(report[lab]["f1-score"]), 4),
                "support": int(report[lab]["support"]),
            }
            for lab in present
        },
        "confusion_matrix_labels": present,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=present).tolist(),
        "n_samples": int(len(y_true)),
    }


def train_and_evaluate(df: pd.DataFrame, target_col: str, labels: list,
                       task_name: str, random_state: int = RANDOM_STATE) -> dict:
    """
    Treina no avalia model hotu ba tarefa ida (sentiment ka kategoria).

    Retorna dict ho: status, class_distribution, per-model holdout metrics,
    cross-validation scores, best_model, no predisaun test-set.
    """
    usable = df[
        df["has_text"]
        & df[TEXT_COL].fillna("").str.len().gt(0)
        & df[target_col].notna()
    ].copy()

    if len(usable) < 20:
        return {"status": "insufficient_data", "task": task_name, "n": int(len(usable))}

    counts = usable[target_col].astype(str).value_counts()
    # Klase ho membru < 2 la bele entra iha split/CV estratifikadu
    valid_classes = counts[counts >= 2].index.tolist()
    dropped = counts[counts < 2].to_dict()
    usable = usable[usable[target_col].astype(str).isin(valid_classes)]

    if usable[target_col].nunique() < 2:
        return {"status": "single_class_only", "task": task_name}

    grouped = target_col == "category_label"
    if grouped:
        # ==================================================================
        # DEMO DOSEN 2A — TRAIN/TEST KATEGORI
        # Split berdasarkan place_id. Semua review dari satu destinasi harus
        # berada di train ATAU test agar tidak terjadi data leakage.
        # ==================================================================
        # A category belongs to a destination. Keep every review of a place in
        # one partition. A class with only one place cannot have an independent
        # test example; retain it in training and report the missing coverage.
        places = usable[["place_id", target_col]].drop_duplicates()
        if places["place_id"].duplicated().any():
            raise ValueError("Conflicting category labels for one destination")
        group_counts = places[target_col].value_counts()
        eligible = places[places[target_col].map(group_counts).ge(2)]
        singleton = places[places[target_col].map(group_counts).lt(2)]
        if len(eligible) < 4:
            return {"status": "insufficient_independent_places", "task": task_name}
        test_size = max(int(np.ceil(len(eligible) * 0.25)), eligible[target_col].nunique())
        tr_places, te_places = train_test_split(
            eligible, test_size=test_size, random_state=random_state,
            stratify=eligible[target_col],
        )
        train_ids = set(tr_places["place_id"]) | set(singleton["place_id"])
        train = usable[usable["place_id"].isin(train_ids)]
        test = usable[usable["place_id"].isin(te_places["place_id"])]
    else:
        # ==================================================================
        # DEMO DOSEN 2B — TRAIN/TEST SENTIMENT
        # Split 75% train dan 25% test dengan stratifikasi label sentiment.
        # ==================================================================
        tr_idx, te_idx = train_test_split(
            usable.index, test_size=0.25, random_state=random_state,
            stratify=usable[target_col].astype(str),
        )
        train, test = usable.loc[tr_idx], usable.loc[te_idx]

    X_train, y_train = train[TEXT_COL], train[target_col].astype(str)
    X_test, y_test = test[TEXT_COL], test[target_col].astype(str)

    n_folds = int(min(CV_FOLDS, train["place_id"].nunique() if grouped
                      else y_train.value_counts().min()))
    cv = (GroupKFold(n_splits=n_folds) if grouped else
          StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)) if n_folds >= 2 else None
    results = {
        "status": "ok",
        "task": task_name,
        "target_column": target_col,
        "n_total": int(len(usable)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "class_distribution": usable[target_col].astype(str).value_counts().to_dict(),
        "classes_dropped_too_few": dropped,
        "cv_folds_used": n_folds if n_folds >= 2 else None,
        "split_method": "stratified_place_holdout" if grouped else "stratified_review_holdout",
        "cv_scope": "training_partition_only",
        "cv_method": "GroupKFold(place_id)" if grouped else "StratifiedKFold",
        "train_class_distribution": y_train.value_counts().to_dict(),
        "test_class_distribution": y_test.value_counts().to_dict(),
        "untested_classes": [lab for lab in labels if lab not in set(y_test)],
        "place_overlap_train_test": len(set(train["place_id"]) & set(test["place_id"])),
        "limitation": (
            "Category holdout separates destinations. Classes absent from test are not validated; "
            "CV folds may lack rare classes. F1-macro uses all configured classes."
            if grouped else "Review holdout estimates performance on reviews; destinations may occur in both partitions."
        ),
        "models": {},
    }

    predictions = {}
    fitted_models = {}
    for name, model in _make_models(random_state).items():
        # ==================================================================
        # DEMO DOSEN 3 — TRAINING DAN TESTING SETIAP MODEL
        # fit()    : model belajar menggunakan X_train dan y_train.
        # predict(): model menjawab data X_test yang tidak dipakai belajar.
        # ==================================================================
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        block = _metric_block(y_test, y_pred, labels)

        if cv is not None:
            # ==============================================================
            # DEMO DOSEN 4 — CROSS-VALIDATION PADA TRAIN SAJA
            # CV F1-macro dipakai untuk memilih model secara adil ketika
            # jumlah anggota setiap kelas tidak seimbang.
            # ==============================================================
            scores = cross_val_score(
                model, X_train, y_train,
                groups=train["place_id"] if grouped else None,
                cv=cv, scoring=lambda estimator, X, y: f1_score(
                    y, estimator.predict(X), labels=labels, average="macro", zero_division=0),
                error_score="raise",
            )
            block["cv_f1_macro_mean"] = round(float(np.mean(scores)), 4)
            block["cv_f1_macro_std"] = round(float(np.std(scores)), 4)

        results["models"][name] = block
        predictions[name] = pd.Series(y_pred, index=test.index)
        fitted_models[name] = model

    # ======================================================================
    # DEMO DOSEN 5 — PEMILIHAN MODEL TERBAIK
    # Model dipilih dari CV F1-macro tertinggi pada TRAIN. Nilai TEST tidak
    # digunakan untuk memilih model; test hanya untuk evaluasi akhir.
    # ======================================================================
    # Test scores and predicted-class counts must never select the model.
    best = (max(results["models"], key=lambda name: results["models"][name]["cv_f1_macro_mean"])
            if cv is not None else "Logistic Regression")
    results["best_model"] = best
    results["best_f1_macro"] = results["models"][best]["f1_macro"]
    results["best_cv_f1_macro"] = results["models"][best].get("cv_f1_macro_mean")
    results["selection_rule"] = (
        "Select the model using cross-validated macro F1 on training data only. "
        "If cross-validation is unavailable, use Logistic Regression as the predefined fallback. "
        "The test set is not used for model selection."
    )
    results["degenerate_models"] = [
        m for m, b in results["models"].items() if b.get("is_degenerate")
    ]
    results["_best_estimator"] = fitted_models[best]
    results["_predictions"] = predictions
    results["_test_index"] = test.index
    results["_train_index"] = train.index
    return results


def run_supervised_models(df: pd.DataFrame, save_models: bool = False) -> tuple:
    """
    Halao tarefa klasifikasaun rua (sentiment + kategoria) no hatama predisaun
    model di'ak liu ba DataFrame:
        sentiment_pred_ml  /  category_pred_ml   (deit ba baris test-set)

    Se save_models=True, grava pipeline model di'ak liu (TF-IDF + classifier)
    ba models/. Model ne'e sei treinadu deit ho partisaun train.
    """
    df = df.copy()

    sentiment_res = train_and_evaluate(
        df, target_col="sentiment_true_proxy", labels=SENTIMENTS,
        task_name="Sentiment Classification (3 klase)",
    )
    category_res = train_and_evaluate(
        df, target_col="category_label", labels=CATEGORIES,
        task_name="Category Classification (4 categories)",
    )

    if save_models:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for res, artifact_path in (
            (sentiment_res, SENTIMENT_MODEL_PATH),
            (category_res, CATEGORY_MODEL_PATH),
        ):
            if res.get("status") == "ok":
                joblib.dump(res["_best_estimator"], artifact_path)
                res["model_artifact"] = str(artifact_path)
            else:
                res["model_artifact"] = None

    for res, col in ((sentiment_res, "sentiment_pred_ml"),
                     (category_res, "category_pred_ml")):
        df[col] = None
        split_col = col.replace("_pred_ml", "_split")
        df[split_col] = "not_used"
        if res.get("status") == "ok":
            preds = res["_predictions"][res["best_model"]]
            df.loc[preds.index, col] = preds.values
            df.loc[res["_train_index"], split_col] = "train"
            df.loc[res["_test_index"], split_col] = "test"

    # Hasai objetu internu antes serializa ba JSON
    for res in (sentiment_res, category_res):
        res.pop("_predictions", None)
        res.pop("_test_index", None)
        res.pop("_train_index", None)
        res.pop("_best_estimator", None)

    return df, {"sentiment": sentiment_res, "category": category_res}


if __name__ == "__main__":
    from category_classifier import apply_category_classification
    from data_collection import collect
    from preprocessing import preprocess
    from sentiment_analysis import apply_sentiment_analysis

    task_arg = "all"
    if "--task" in sys.argv:
        try:
            task_arg = sys.argv[sys.argv.index("--task") + 1].lower()
        except IndexError:
            raise SystemExit("Gunakan: --task sentiment atau --task category")
    if task_arg not in {"all", "sentiment", "category"}:
        raise SystemExit("Task tidak dikenal. Pilih: sentiment atau category")

    # Standalone demo mode: prepare the same in-memory data used by the official
    # pipeline, without overwriting the generated CSV/JSON outputs.
    raw = collect(write_log=False)
    clean, _ = preprocess(raw)
    d = apply_category_classification(clean)
    d = apply_sentiment_analysis(d)

    res = {}
    if task_arg in {"all", "sentiment"}:
        res["sentiment"] = train_and_evaluate(
            d,
            target_col="sentiment_true_proxy",
            labels=SENTIMENTS,
            task_name="Sentiment Classification (3 classes)",
        )
    if task_arg in {"all", "category"}:
        res["category"] = train_and_evaluate(
            d,
            target_col="category_label",
            labels=CATEGORIES,
            task_name="Category Classification (4 categories)",
        )

    for task, r in res.items():
        print("=" * 70)
        print(f"TAREFA: {task}  |  status={r.get('status')}")
        if r.get("status") != "ok":
            continue
        print(f"train={r['n_train']} test={r['n_test']} cv_folds={r['cv_folds_used']}")
        print(f"distribuisaun klase: {r['class_distribution']}")
        rows = [{
            "Model": m,
            "Accuracy": v["accuracy"],
            "Precision(macro)": v["precision_macro"],
            "Recall(macro)": v["recall_macro"],
            "F1(macro)": v["f1_macro"],
            "CV F1": v.get("cv_f1_macro_mean"),
        } for m, v in r["models"].items()]
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"-> Model di'ak liu: {r['best_model']} (F1 macro = {r['best_f1_macro']})")
