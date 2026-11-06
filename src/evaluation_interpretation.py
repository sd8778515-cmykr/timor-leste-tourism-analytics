"""Evidence-based interpretation shared by dashboard and pipeline output."""
from pathlib import Path


def build_interpretation(evaluation: dict) -> dict:
    tasks = {}
    for key, title in (("sentiment", "Sentiment"), ("category", "Category")):
        task = evaluation[key]
        ml = task.get("model_2_supervised", {})
        baseline = task.get("majority_baseline", {})
        if ml.get("status") != "ok" or baseline.get("status") != "ok":
            tasks[key] = {"title": title, "status": "insufficient_data",
                          "summary": "There is not enough data to compare the model with the baseline."}
            continue
        selected = ml["models"][ml["best_model"]]
        delta = round(selected["f1_macro"] - baseline["f1_macro"], 4)
        per_class = selected["per_class"]
        supported = {lab: block for lab, block in per_class.items() if block["support"] > 0}
        weakest = min(supported, key=lambda lab: supported[lab]["f1_score"])
        labels = selected["confusion_matrix_labels"]
        mistakes = sorted([
            {"actual": a, "predicted": b, "count": selected["confusion_matrix"][i][j]}
            for i, a in enumerate(labels) for j, b in enumerate(labels)
            if i != j and selected["confusion_matrix"][i][j] > 0
        ], key=lambda row: (-row["count"], row["actual"], row["predicted"]))
        comparison = "higher" if delta > 0 else "equal" if delta == 0 else "lower"
        summary = (
            f"{ml['best_model']} was selected using cross-validation on training data only. "
            f"On {ml['n_test']} test records, accuracy={selected['accuracy']:.4f} and "
            f"macro F1={selected['f1_macro']:.4f}. This macro F1 is {comparison} than "
            f"the majority-class baseline ({baseline['f1_macro']:.4f}; difference {delta:+.4f}). "
            "This comparison is descriptive and does not establish statistical significance."
        )
        limits = [
            f"Weakest class represented in the test set: {weakest}, "
            f"F1={supported[weakest]['f1_score']:.4f}, support={supported[weakest]['support']}.",
            "Macro F1 includes every configured class. A zero score for a class with no "
            "test support does not demonstrate performance on that class.",
        ]
        if key == "sentiment":
            limits.append("Reference labels come from star ratings rather than human text annotations. "
                          "Evaluation therefore measures agreement with a rating proxy. "
                          "The same destination may occur in training and test data.")
        else:
            limits.append("Training and test sets are separated by destination. Reference labels "
                          "are limited, so rare-class results are sensitive to the split.")
        if ml.get("untested_classes"):
            limits.append("Not represented in the independent test set: " + ", ".join(ml["untested_classes"]) + ".")
        tasks[key] = {
            "title": title, "status": "ok", "selected_model": ml["best_model"],
            "summary": summary, "f1_delta_vs_majority": delta,
            "baseline_class": baseline["majority_class"],
            "limitations": limits, "largest_confusions": mistakes[:5],
        }
    cluster = evaluation.get("clustering", {})
    metrics = cluster.get("metrics", {})
    if metrics.get("status") == "ok":
        sil = metrics["silhouette_score"]
        separation = ("weak separation with substantial overlap" if sil < 0.2
                      else "separation should be interpreted with cluster profiles and sizes")
        # A descriptive caution, not a universal pass/fail threshold.
        note = (
            f"K-Means with k={metrics['k_used']} produced silhouette={sil:.4f}, indicating {separation}. "
            f"The highest silhouette in the tested range occurred at k={metrics['k_best_by_silhouette']}. "
            "The configured k is an analytical choice rather than an automatically optimal value. "
            f"The 2D SVD projection retains {100 * metrics['svd_explained_variance_total']:.2f}% of variance; "
            "the scatter plot is illustrative. Text clusters are not equivalent to tourism categories."
        )
    else:
        note = "There is not enough data to interpret clustering."
    return {"tasks": tasks, "clustering": note,
            "scope": "Evaluation uses the full dataset and remains fixed when dashboard filters change."}


def write_interpretation(evaluation: dict, path: Path) -> None:
    """Regenerate a short audit note from exactly the data shown in the UI."""
    info = evaluation["interpretation"]
    lines = ["# Evaluasi dan interpretasi sistem", "", info["scope"], ""]
    for task in info["tasks"].values():
        lines += [f"## {task['title']}", "", task["summary"], ""]
        lines += [f"- {note}" for note in task.get("limitations", [])]
        lines += ["", "Kesalahan terbanyak pada model terpilih:", ""]
        lines += [f"- {row['actual']} → {row['predicted']}: {row['count']} review."
                  for row in task.get("largest_confusions", [])]
        lines += [""]
    lines += ["## Clustering", "", info["clustering"], "", "## Validasi manusia", ""]
    manual = evaluation["manual_validation"]
    for key in ("sentiment", "category"):
        item = manual["results"][key]
        lines.append(f"- {key}: {item['status']}; {item['n_valid']} label valid dari {manual['n_sample']} sampel; "
                     f"{item['n_invalid']} label tidak dikenali, {item['n_unmatched']} review tidak cocok dengan dataset.")
    lines += ["", "Isi `reports/tables/sample_validasaun_manual.csv` berdasarkan penilaian manusia. "
              "Label yang kosong tidak diisi otomatis. Jalankan pipeline ulang untuk menghitung hasil validasi.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
