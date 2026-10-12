"""Matched five-stage Random Forest / neural-network history experiment.

Population models exclude every subject in the outer test fold. Source subjects'
last nights select model settings; earlier nights train. Target subjects use
their final two nights for validation/test and k earlier nights for adaptation.
No test score chooses parameters, history, stopping, or probability smoothing.

Personalization mode "anchored" (default) keeps each target subject's
validation and test nights fixed while the k training nights grow backward.
Mode "walk_forward" advances the split with history: train nights 1..k,
validate night k+1, test night k+2, mirroring live device learning.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

LABELS = np.arange(5)
STAGES = ["wake", "n1", "n2", "n3", "rem"]
BASE_FEATURES = [
    "HR", "hr_std", "hr_min", "hr_max", "hr_range", "hr_count", "hr_change_prev",
    "enmo_mean", "enmo_std", "enmo_min", "enmo_max", "enmo_range",
]


def temporal_features(raw):
    """Causal sensor context; never cross nights or compress unknown epochs."""
    required = {"subject", "night", "epoch", "sleep_stage", *BASE_FEATURES}
    if required - set(raw):
        raise ValueError(f"Missing input columns: {sorted(required - set(raw))}")
    data = raw.copy()
    data["subject"] = data.subject.astype(str)
    data["night"] = pd.to_numeric(data.night, errors="raise").astype(int)
    if data.duplicated(["subject", "night", "epoch"]).any():
        raise ValueError("Duplicate subject/night/epoch")
    frames = []
    for _, night in data.groupby(["subject", "night"], sort=True):
        night = night.sort_values("epoch").copy()
        if not np.all(np.diff(night.epoch) == 1):
            raise ValueError("Input must retain consecutive epochs, including unknown labels")
        night["elapsed_hours"] = night.epoch / 120.0
        for name in ["HR", "hr_count", "enmo_mean", "enmo_std", "enmo_max"]:
            values = night[name].astype(float).replace([np.inf, -np.inf], np.nan)
            if name.startswith("enmo"):
                values = np.log1p(values.clip(lower=0) * 1000)
            night[f"{name}_missing"] = values.isna().astype(float)
            for width in [2, 10, 30]:
                roll = values.rolling(width, min_periods=1)
                night[f"{name}_mean_{width}"] = roll.mean()
                night[f"{name}_std_{width}"] = roll.std(ddof=0)
            night[f"{name}_delta_10"] = values - values.shift(10)
        hr = night.HR.astype(float)
        night["hr_relative"] = hr - hr.expanding(min_periods=1).median()
        frames.append(night)
    result = pd.concat(frames, ignore_index=True)
    features = BASE_FEATURES + [c for c in result if c not in data.columns]
    result[features] = result[features].replace([np.inf, -np.inf], np.nan)
    return result[result.sleep_stage.isin(LABELS)].reset_index(drop=True), features


def make_target_splits(data, subject, history, mode="anchored"):
    person = data[data.subject == subject]
    nights = sorted(person.night.unique())
    if mode == "walk_forward":
        if len(nights) < 3 or history > len(nights) - 2:
            raise ValueError("Insufficient nights")
        train_nights = nights[:history] if history else []
        validation_night = nights[history]
        test_night = nights[history + 1]
    else:
        if len(nights) < 3 or history > len(nights) - 2:
            raise ValueError("Insufficient nights")
        train_nights = nights[:-2][-history:] if history else []
        validation_night = nights[-2]
        test_night = nights[-1]
    parts = [person[person.night.isin(train_nights)],
             person[person.night == validation_night], person[person.night == test_night]]
    return parts, dict(subject=subject, history=history, mode=mode,
                       train_nights=list(map(int, train_nights)),
                       validation_night=int(validation_night), test_night=int(test_night))


def probabilities(model, x):
    result = np.full((len(x), 5), 1e-8)
    result[:, np.asarray(model.classes_, dtype=int)] += model.predict_proba(x)
    return result / result.sum(axis=1, keepdims=True)


def scores(y, p):
    pred = p.argmax(axis=1)
    matrix = confusion_matrix(y, pred, labels=LABELS)
    support = matrix.sum(axis=1)
    recall = np.divide(matrix.diagonal(), support, out=np.zeros(5), where=support > 0)
    result = dict(accuracy=float(accuracy_score(y, pred)),
                  macro_f1=float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
                  balanced_accuracy=float(recall[support > 0].mean()),
                  log_loss=float(log_loss(y, p, labels=LABELS)))
    result.update(zip([f"f1_{s}" for s in STAGES],
                      map(float, f1_score(y, pred, labels=LABELS, average=None, zero_division=0))))
    return result


def mean_subject_f1(frame, p):
    return float(np.mean([scores(frame.sleep_stage.to_numpy()[indices], p[indices])["macro_f1"]
                          for indices in frame.reset_index(drop=True).groupby("subject").indices.values()]))


def weights(frame):
    """Equal subject influence plus mild (sqrt inverse frequency) class balancing."""
    y = frame.sleep_stage.to_numpy(dtype=int)
    counts = np.bincount(y, minlength=5)
    class_weight = np.sqrt(len(y) / (5 * np.maximum(counts, 1)))
    subject_count = frame.subject.map(frame.subject.value_counts()).to_numpy()
    w = class_weight[y] / subject_count
    return w / w.mean()


def transitions(frame):
    counts = np.ones((5, 5), dtype=float)
    for _, night in frame.groupby(["subject", "night"]):
        y = night.sleep_stage.to_numpy(dtype=int)
        consecutive = np.diff(night.epoch) == 1
        np.add.at(counts, (y[:-1][consecutive], y[1:][consecutive]), 1)
    return counts / counts.sum(axis=1, keepdims=True)


def filter_probabilities(p, frame, transition, strength):
    """Forward-only Markov filtering; reset at nights and missing-label gaps."""
    if strength == 0:
        return p
    out = p.copy()
    epochs = frame.epoch.to_numpy()
    nights = list(zip(frame.subject, frame.night))
    for i in range(1, len(p)):
        if nights[i] == nights[i - 1] and epochs[i] == epochs[i - 1] + 1:
            prior = out[i - 1] @ transition
            out[i] *= np.maximum(prior, 1e-8) ** strength
            out[i] /= out[i].sum()
    return out


def batch_iterator(values, labels, batch_weight, rng, batch_size):
    """Yield shuffled (x_batch, y_batch, w_batch) mini-batches forever."""
    while True:
        order = rng.permutation(len(values))
        for start in range(0, len(values), batch_size):
            pick = order[start:start + batch_size]
            yield values[pick], labels[pick], batch_weight[pick]


class MiniBatchMLP:
    """Two-hidden-layer MLP wrapper exposing per-batch training loss logging.

    Mirrors the previous MLPClassifier configuration: Adam, alpha=0.01,
    learning_rate 0.001, relu activations, softmax output.
    """

    def __init__(self, hidden, seed, alpha=0.01):
        self.hidden = hidden
        self.seed = seed
        self.alpha = alpha
        self.layers = None
        self.optimizer = None

    def _init(self, x, w):
        sizes = [x.shape[1], *self.hidden, 5]
        rng = np.random.default_rng(self.seed)
        self.layers = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            bound = np.sqrt(6.0 / (fan_in + fan_out))
            self.layers.append({
                "w": rng.uniform(-bound, bound, size=(fan_in, fan_out)).astype(np.float32),
                "b": np.zeros(fan_out, dtype=np.float32),
            })
        self.optimizer = [{"m": np.zeros_like(l["w"]), "v": np.zeros_like(l["w"]),
                           "mb": np.zeros_like(l["b"]), "vb": np.zeros_like(l["b"]),
                           "t": 0} for l in self.layers]

    def _forward(self, x):
        activations = [x]
        for layer in self.layers[:-1]:
            z = activations[-1] @ layer["w"] + layer["b"]
            activations.append(np.maximum(z, 0))
        last = self.layers[-1]
        logits = activations[-1] @ last["w"] + last["b"]
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return activations, probs

    def _adam(self, index, grad_w, grad_b, lr, weight_decay=1e-4):
        state = self.optimizer[index]
        state["t"] += 1
        layer = self.layers[index]
        for param, grad, key in [(layer["w"], grad_w, "m"), (layer["b"], grad_b, "mb")]:
            pass
        state["m"] = 0.9 * state["m"] + 0.1 * grad_w
        state["v"] = 0.999 * state["v"] + 0.001 * grad_w * grad_w
        m_hat = state["m"] / (1 - 0.9 ** state["t"])
        v_hat = state["v"] / (1 - 0.999 ** state["t"])
        layer["w"] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8) + lr * weight_decay * layer["w"]
        state["mb"] = 0.9 * state["mb"] + 0.1 * grad_b
        state["vb"] = 0.999 * state["vb"] + 0.001 * grad_b * grad_b
        state["tb"] = state["t"]
        m_b = state["mb"] / (1 - 0.9 ** state["t"])
        v_b = state["vb"] / (1 - 0.999 ** state["t"])
        layer["b"] -= lr * m_b / (np.sqrt(v_b) + 1e-8)

    def train_batches(self, values, labels, batch_weight, classes, rng, batch_size,
                      batches_per_epoch, lr=0.001):
        """Run one epoch of mini-batches; return per-batch weighted loss."""
        if self.layers is None:
            self._init(values, batch_weight)
        batch_weight = np.asarray(batch_weight, dtype=np.float32)
        labels = np.asarray(labels)
        losses = []
        stream = batch_iterator(values, labels, batch_weight, rng, batch_size)
        for _ in range(batches_per_epoch):
            xb, yb, wb = next(stream)
            activations, probs = self._forward(xb)
            loss = float(-np.sum(np.log(np.maximum(probs[np.arange(len(yb)), yb], 1e-12)) * wb))
            losses.append(loss)
            probs[np.arange(len(yb)), yb] -= 1.0
            probs *= wb[:, None]
            delta = probs / len(yb)
            for index in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[index]
                grad_w = activations[index].T @ delta
                grad_b = delta.sum(axis=0)
                self._adam(index, grad_w, grad_b, lr)
                if index:
                    delta = (delta @ layer["w"].T) * (activations[index] > 0)
        return losses

    def predict_proba(self, values):
        values = np.asarray(values, dtype=np.float32)
        _, probs = self._forward(values)
        return probs

    def classes_(self):
        return np.arange(5)

    classes_ = property(classes_)


def fit_population(train, val, features, args):
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    xt = scaler.fit_transform(imp.fit_transform(train[features])).astype(np.float32)
    xv = scaler.transform(imp.transform(val[features])).astype(np.float32)
    y = train.sleep_stage.to_numpy(dtype=int)
    w = weights(train)
    selected, search = {}, []
    best_score = -1
    for leaf, depth in [(5, None), (20, 18)]:
        model = RandomForestClassifier(n_estimators=args.trees, min_samples_leaf=leaf,
                    max_depth=depth, max_features="sqrt", n_jobs=args.jobs, random_state=args.seed)
        model.fit(xt, y, sample_weight=w)
        score = mean_subject_f1(val, probabilities(model, xv))
        print(f"  RF leaf={leaf}, depth={depth}: source-validation macro-F1={score:.3f}", flush=True)
        search.append(dict(model="random_forest", leaf=leaf, depth=depth, macro_f1=score))
        if score > best_score:
            best_score, selected["random_forest"] = score, model
    best_score = -1
    for hidden in [(64, 32), (128, 64)]:
        batch_size = min(256, len(train))
        batches_per_epoch = int(np.ceil(len(train) / batch_size))
        model = MiniBatchMLP(hidden, args.seed)
        rng = np.random.default_rng(args.seed)
        stale, local_best = 0, -1
        for epoch in range(1, args.epochs + 1):
            losses = model.train_batches(xt, y, w, LABELS, rng, batch_size, batches_per_epoch)
            score = mean_subject_f1(val, probabilities(model, xv))
            for batch, loss in enumerate(losses, start=1):
                search.append(dict(model="neural_network", hidden=hidden, epoch=epoch,
                                   batch=batch, global_batch=(epoch - 1) * batches_per_epoch + batch,
                                   train_loss=loss, macro_f1=None))
            search.append(dict(model="neural_network", hidden=hidden, epoch=epoch,
                               batch=batches_per_epoch,
                               global_batch=(epoch - 1) * batches_per_epoch + batches_per_epoch,
                               train_loss=losses[-1] if losses else None, macro_f1=score))
            if score > best_score:
                best_score, selected["neural_network"] = score, deepcopy(model)
            if score > local_best + 0.001:
                local_best, stale = score, 0
            else:
                stale += 1
            if stale >= 5:
                break
        print(f"  MLP hidden={hidden}: best source-validation macro-F1={local_best:.3f}, epochs={epoch}", flush=True)
    return selected, imp, scaler, search


def bootstrap(values, seed=42):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return list(map(float, np.quantile(sampled, [0.025, 0.975])))


def subject_equal_mean(frame, metric):
    """Average each subject's rows first, then average subjects equally."""
    return float(frame.groupby("subject")[metric].mean().mean())


def subject_equal_bootstrap(frame, metric, seed=42):
    """Bootstrap over per-subject means so people, not rows, are sampled."""
    subject_means = frame.groupby("subject")[metric].mean().to_numpy(dtype=float)
    return bootstrap(subject_means, seed)


def write_report(rows, output, mode):
    results = pd.DataFrame(rows)
    results.to_csv(output / "metrics.csv", index=False)
    test = results[results.split == "test"]
    common = test.groupby("subject").history.max()
    common = common[common >= 5].index
    max_k = test.groupby("subject").history.transform("max")
    # Main cohort: every personalized history k >= 1, averaged per subject first
    # so long-history people do not dominate. Unlike the previous max-history-only
    # cohort, this differs between anchored and walk_forward modes.
    cohorts = {"all_subjects_personalized_k1_plus": test[test.history >= 1]}
    cohorts["all_subjects_max_available_history"] = test[test.history == max_k]
    if len(common):
        for k in range(6):
            cohorts[f"common_cohort_k{k}"] = test[test.subject.isin(common) & (test.history == k)]
    summary, paired = [], []
    metric_names = ["accuracy", "macro_f1", "balanced_accuracy", "log_loss", *[f"f1_{s}" for s in STAGES]]
    for cohort, frame in cohorts.items():
        for model, group in frame.groupby("model"):
            row = dict(cohort=cohort, model=model, subjects=group.subject.nunique())
            for metric in metric_names:
                if cohort == "all_subjects_personalized_k1_plus":
                    row[metric] = subject_equal_mean(group, metric)
                    row[metric + "_ci95"] = subject_equal_bootstrap(group, metric)
                else:
                    row[metric] = float(group[metric].mean())
                    row[metric + "_ci95"] = bootstrap(group[metric])
            summary.append(row)
        for metric in ["accuracy", "macro_f1"]:
            pivot = frame.pivot_table(index="subject", columns="model", values=metric,
                                      aggfunc="mean")
            delta = pivot.neural_network - pivot.random_forest
            paired.append(dict(cohort=cohort, metric=metric, difference_nn_minus_rf=float(delta.mean()),
                               ci95=bootstrap(delta), subjects=len(delta)))
            if "original_rf" in pivot:
                for model in ["random_forest", "neural_network"]:
                    improvement = pivot[model] - pivot.original_rf
                    paired.append(dict(cohort=cohort, metric=metric, model=model,
                                       reference="original_rf", mean_difference=float(improvement.mean()),
                                       ci95=bootstrap(improvement), subjects=len(improvement)))
    (output / "summary.json").write_text(json.dumps(dict(mode=mode, summary=summary, paired=paired), indent=2))
    lines = ["# Sleep model comparison", "", "Five stages: Wake, N1, N2, N3, REM. All percentages below are measured on final held-out nights.", "",
             f"Personalization mode: **{mode}**. Anchored keeps each target person's validation and test nights fixed while history grows backward; walk_forward trains on nights 1..k, validates night k+1, and tests night k+2, mirroring live device learning.", "",
             "Population training excludes the target subject's outer fold. Both models use identical sensor features and source/target splits. Source validation chooses model settings and neural-network stopping. Target validation chooses personalization weight and causal smoothing. History k counts training nights; one additional labeled validation night is required. k=0 is a population-only baseline, with no target validation tuning.", "",
             "Temporal features use current and preceding 1/5/15-minute windows and elapsed time. No future sensor values or true prior stages enter predictions. A compact two-layer MLP is the neural comparator; this is not a reproduction of the published LSTM.", "",
             "| Cohort | Model | People | Accuracy | Macro-F1 |", "|---|---|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['cohort']} | {row['model']} | {row['subjects']} | {row['accuracy']:.1%} | {row['macro_f1']:.3f} |")
    lines += ["", "The original RF baseline is retrained on exactly the same target training nights, using its original features/settings; it has no population training. Its contrast with the new models measures the entire pipeline change, not just the classifier. The two new models share all available training data.", "",
              "The main cohort (all_subjects_personalized_k1_plus) averages every personalized history k>=1, giving each person equal weight: per-subject means first, then the subject mean. This cohort distinguishes anchored from walk_forward evaluation; the max-history cohort is identical in both because splits coincide at k=max. Only the common cohort supports a controlled 0-5-night curve. Participants, not epochs, are the statistical units. Test results must not be used to retune this experiment. No clinical reliability threshold or minimum-night claim is established."]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    raw = pd.read_csv(args.input)
    data, features = temporal_features(raw)
    quality = raw.groupby(["subject", "night"])[BASE_FEATURES].agg(lambda s: s.isna().mean())
    quality.to_csv(output / "missing_sensor_fraction.csv")
    raw.groupby(["subject", "night", "sleep_stage"]).size().rename("epochs").to_csv(
        output / "stage_counts.csv")
    subjects = np.array(sorted(data.subject.unique()))
    if any(data[data.subject == s].night.nunique() < 3 for s in subjects):
        raise ValueError("All subjects need at least three nights")
    config = vars(args).copy()
    config.update(input=str(args.input), output=str(args.output), features=features,
                  input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  sklearn=sklearn.__version__, python=platform.python_version())
    (output / "configuration.json").write_text(json.dumps(config, indent=2))
    rows, manifest, search_rows, prediction_frames = [], [], [], []
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (source_idx, target_idx) in enumerate(splitter.split(subjects), 1):
        source = data[data.subject.isin(subjects[source_idx])]
        final = source.groupby("subject").night.transform("max")
        train, val = source[source.night < final], source[source.night == final]
        source_manifest = dict(fold=fold, source_train=sorted(set(zip(train.subject, map(int, train.night)))),
                               source_validation=sorted(set(zip(val.subject, map(int, val.night)))),
                               target_subjects=subjects[target_idx].tolist())
        print(f"Fold {fold}/{args.folds}: population training ({len(train)} epochs)", flush=True)
        models, imp, scaler, searches = fit_population(train, val, features, args)
        search_rows.extend([dict(fold=fold, **row) for row in searches])
        transition = transitions(train)
        source_val_x = scaler.transform(imp.transform(val[features])).astype(np.float32)
        source_strength = {}
        for name, model in models.items():
            p = probabilities(model, source_val_x)
            source_strength[name] = max([0.0, 0.5, 1.0], key=lambda strength:
                mean_subject_f1(val, filter_probabilities(p, val, transition, strength)))
        for subject in subjects[target_idx]:
            print(f"  Adapting both models for {subject}", flush=True)
            max_history = min(5, data[data.subject == subject].night.nunique() - 2)
            for k in range(max_history + 1):
                parts, split_manifest = make_target_splits(data, subject, k, args.mode)
                personal, validation, test = parts
                manifest.append(dict(**source_manifest, **split_manifest))
                arrays = [scaler.transform(imp.transform(p[features])).astype(np.float32)
                          if len(p) else None for p in parts]
                xt, xv, xe = arrays
                for name, population in models.items():
                    pop_v = probabilities(population, xv)
                    local = None
                    if k:
                        if name == "random_forest":
                            local = clone(population)
                            local.fit(xt, personal.sleep_stage, sample_weight=weights(personal))
                        else:
                            local = deepcopy(population)
                            # Fresh shuffling stream and a conservative learning rate
                            # keep personal fine-tuning small and deterministic.
                            adapt_rng = np.random.default_rng(args.seed + 1)
                            adapt_batch = min(256, len(personal))
                            for _ in range(args.adapt_epochs):
                                local.train_batches(xt, personal.sleep_stage, weights(personal),
                                                     LABELS, adapt_rng, adapt_batch,
                                                     max(1, int(np.ceil(len(personal) / adapt_batch))),
                                                     lr=0.0001)
                    local_v = probabilities(local, xv) if local else pop_v
                    best = (-1, 0.0, source_strength[name])
                    if k:
                        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
                            mixed = (1-alpha)*pop_v + alpha*local_v
                            for strength in [0.0, 0.5, 1.0]:
                                score = scores(validation.sleep_stage,
                                    filter_probabilities(mixed, validation, transition, strength))["macro_f1"]
                                if score > best[0]:
                                    best = (score, alpha, strength)
                    _, alpha, strength = best
                    for split, frame, x in zip(["train", "validation", "test"], parts, arrays):
                        if not len(frame):
                            continue
                        p = probabilities(population, x)
                        if local and alpha:
                            p = (1-alpha)*p + alpha*probabilities(local, x)
                        p = filter_probabilities(p, frame, transition, strength)
                        rows.append(dict(fold=fold, subject=subject, history=k, model=name,
                                         split=split, epochs=len(frame), alpha=alpha, smoothing=strength,
                                         **scores(frame.sleep_stage, p)))
                        if split == "test":
                            pred = frame[["subject", "night", "epoch", "sleep_stage"]].copy()
                            pred["model"], pred["history"], pred["predicted_stage"] = name, k, p.argmax(axis=1)
                            for j, stage in enumerate(STAGES):
                                pred[f"p_{stage}"] = p[:, j]
                            prediction_frames.append(pred)
                if k:
                    old_imp = SimpleImputer(strategy="median", keep_empty_features=True)
                    bx = old_imp.fit_transform(personal[BASE_FEATURES])
                    baseline = RandomForestClassifier(n_estimators=100, max_depth=12,
                        min_samples_leaf=10, max_features=None, class_weight="balanced_subsample",
                        random_state=args.seed, n_jobs=args.jobs).fit(bx, personal.sleep_stage)
                    for split, frame in zip(["train", "validation", "test"], parts):
                        rows.append(dict(fold=fold, subject=subject, history=k, model="original_rf",
                            split=split, epochs=len(frame), alpha=1.0, smoothing=0.0,
                            **scores(frame.sleep_stage, probabilities(baseline, old_imp.transform(frame[BASE_FEATURES])))))
                    majority = int(personal.sleep_stage.mode().iloc[0])
                    for split, frame in zip(["train", "validation", "test"], parts):
                        p = np.full((len(frame), 5), 1e-8)
                        p[:, majority] = 1
                        p /= p.sum(axis=1, keepdims=True)
                        rows.append(dict(fold=fold, subject=subject, history=k, model="majority_stage",
                            split=split, epochs=len(frame), alpha=1.0, smoothing=0.0,
                            **scores(frame.sleep_stage, p)))
                # Majority-stage accuracy is a more useful reference than 20% uniform guessing.
                if k == max_history:
                    majority = int(personal.sleep_stage.mode().iloc[0])
                    print(f"  {subject}: k={k}, majority test accuracy="
                          f"{np.mean(test.sleep_stage == majority):.3f}", flush=True)
            pd.DataFrame(rows).to_csv(output / "metrics.partial.csv", index=False)
        print(f"Finished fold {fold}, elapsed {time.time()-start:.0f}s", flush=True)
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "validation_search.json").write_text(json.dumps(search_rows, indent=2))
    pd.concat(prediction_frames, ignore_index=True).to_csv(output / "test_predictions.csv", index=False)
    write_report(rows, output, args.mode)
    print(f"Complete: {output / 'SUMMARY.md'} ({time.time()-start:.0f}s)", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("processed_sleep_data_enhanced_hr.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/mode_anchored"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--adapt-epochs", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["anchored", "walk_forward"], default="anchored",
                        help="anchored: fixed target validation/test nights, history grows backward; "
                             "walk_forward: train nights 1..k, validate k+1, test k+2 (live-device style)")
    return parser.parse_args()


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run(parse_args())
