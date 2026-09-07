# -*- coding: utf-8 -*-
"""
STANDALONE SSI RISK-GRADE SCORER
================================

Self-contained companion to grading_model.joblib.

The artefact stores a fitted preprocessor object, and unpickling it requires
that class to be importable. Keeping the class here means the artefact loads in
any fresh Python session, not only the one that produced it.

Usage:
    from ssi_scorer import load_scorer, score_patient
    s = load_scorer("grading_model.joblib")
    print(score_patient({"age_years": 2.0, "OPTIME": 150}, s))

Omitted features are treated as missing and handled by the same imputation the
model was fitted with.

Research tool. It reports the locked model's risk estimate and is not a
substitute for clinical judgement.
"""

import os, json, math, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

# Architecture is read from the Stage-1B handoff; set explicitly only to
# assert which model this stage is allowed to analyse.
LOCKED_MODEL_NAME = "HistGradientBoosting"
BASE_DIR = "."
FALLBACK_DIR = "/mnt/data"







TARGET_COL = "incident_ssi_any"
TIME_COL = "time_to_ssi_or_30d"       # carried only for later survival analysis; never a feature
GROUP_COL = "case_id"
RANDOM_STATE = 20260524
N_CV_FOLDS = 5
OPTIME_NONPOSITIVE_TO_MISSING = True
ADD_CONTINUOUS_MISSING_INDICATORS = True
MISSING_INDICATOR_MIN_RATE = 0.01
AUTO_DOWNLOAD_ZIP = True
CREATE_COLAB_DOWNLOAD_LINK = True
ZIP_COMPRESSION_LEVEL = 1

# Same raw 19 features as Stage 1.
CONTINUOUS_FEATURES = ["age_years", "cefazolin_time_min", "OPTIME"]
BINARY_FEATURES = [
    "sex_binary", "ASA_ge3", "CASETYPE", "ACQ_ABNORMALITY", "IMPCOGSTAT",
    "SEIZURE", "premature_birth_any", "NEUROMUSCDIS", "CEREBRAL_PALSY",
    "NUTR_SUPPORT", "cardiac_comorbidity_any", "STRUCT_PULM_AB"
]
ORDINAL_FEATURES = ["additional_procedure_burden"]
NOMINAL_FEATURES = [
    "procedure_group", "regimen_group", "any_iv_redosed_in_cefazolin_opportunity"
]
ALL_FEATURES = CONTINUOUS_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES
assert len(ALL_FEATURES) == 19

# Thresholds are meaningful for continuous and ordered numeric features, not binary dummies.
# If you want only the three continuous variables, remove additional_procedure_burden here.
SHAP_THRESHOLD_FEATURES = [
    "age_years", "cefazolin_time_min", "OPTIME", "additional_procedure_burden"
]

SHAP_BEESWARM_MAX_DISPLAY = 19
SHAP_BAR_MAX_DISPLAY = 19
SHAP_THRESHOLD_N_BINS = 20
SHAP_THRESHOLD_MIN_BIN_N = 30
SHAP_THRESHOLD_BOOTSTRAPS = 300   # full model-refit bootstrap replicates (see FIX 3)
# Wall-clock cap for the refit bootstrap. Each replicate refits all CV folds
# and recomputes OOF SHAP, so this is the slowest part of the stage. Set to
# None for no cap. Intervals are computed from whatever completed.
SHAP_BOOTSTRAP_MAX_MINUTES = 90
# Multi-transition rule: report ALL stable SHAP sign changes, not only the strongest one.
# A "risk transition" means the smoothed OOF SHAP curve crosses 0 (negative↔positive).
# Tiny opposite-sign excursions are treated as smoothing/model wiggles rather than clinical cutoffs.
SHAP_TRANSITION_MIN_ABS_SIGNAL = 0.005
SHAP_TRANSITION_MIN_RELATIVE_SIGNAL = 0.15
SHAP_TRANSITION_MIN_TEST_N_PER_ADJACENT_REGION = 50
SHAP_TRANSITION_MIN_TEST_EVENTS_PER_ADJACENT_REGION = 3

MISSING_STRINGS = {"", " ", "na", "n/a", "nan", "none", "null", ".", "missing", "<na>"}
ORDINAL_MAPS = {
    "additional_procedure_burden": {
        "0": 0, "1": 1, "2": 2, "3": 3, "3+": 3, ">=3": 3, "3 or more": 3
    }
}
PREFERRED_NOMINAL_LEVELS = {
    "procedure_group": [
        "CSF diversion and shunt procedures",
        "Spine and spinal cord procedures",
        "Functional neurosurgery and neuromodulation procedures",
        "Chiari decompression procedures",
        "Craniofacial reconstruction procedures",
        "Major cranial / intracranial surgery",
    ],
    "regimen_group": ["Cefazolin alone", "Cefazolin + vancomycin", "Other"],
    "any_iv_redosed_in_cefazolin_opportunity": [
        "Not_redosed", "Redosed", "No_redosing_opportunity"
    ],
}
NOMINAL_MISSING_LEVEL = {
    "procedure_group": "Procedure_group_unknown",
    "regimen_group": "Regimen_unknown",
    "any_iv_redosed_in_cefazolin_opportunity": "No_redosing_opportunity",
}
FEATURE_LABELS = {
    "age_years": "Age (years)",
    "cefazolin_time_min": "Cefazolin timing relative to incision (min)",
    "OPTIME": "Operative time (min)",
    "sex_binary": "Sex (male)",
    "ASA_ge3": "ASA physical status >= 3",
    "CASETYPE": "Case type (emergency/urgent)",
    "ACQ_ABNORMALITY": "Acquired CNS abnormality",
    "IMPCOGSTAT": "Impaired cognitive status",
    "SEIZURE": "Seizure disorder",
    "premature_birth_any": "Premature birth",
    "NEUROMUSCDIS": "Neuromuscular disorder",
    "CEREBRAL_PALSY": "Cerebral palsy",
    "NUTR_SUPPORT": "Nutritional support",
    "cardiac_comorbidity_any": "Any cardiac comorbidity",
    "STRUCT_PULM_AB": "Structural pulmonary abnormality",
    "additional_procedure_burden": "Additional procedure burden",
    "procedure_group": "Procedure group",
    "regimen_group": "Antibiotic regimen group",
    "any_iv_redosed_in_cefazolin_opportunity": "IV redosing within cefazolin opportunity",
}


def clean_scalar(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        s = x.strip().replace("≥", ">=")
        return np.nan if s.lower() in MISSING_STRINGS else s
    return x


def norm_text(x):
    x = clean_scalar(x)
    if pd.isna(x):
        return None
    return str(x).strip().replace("≥", ">=").lower()


def to_binary_target(x):
    sx = norm_text(x)
    if sx is None:
        return np.nan
    if sx in {"1", "1.0", "yes", "y", "true", "t"}:
        return 1.0
    if sx in {"0", "0.0", "no", "n", "false", "f"}:
        return 0.0
    try:
        v = float(sx)
        return float(v) if v in (0.0, 1.0) else np.nan
    except Exception:
        return np.nan


def pretty_feature_name(f):
    return FEATURE_LABELS.get(f, f.replace("_", " "))


def safe_filename(x):
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))).strip("_")[:180] or "file"


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_member(zf, suffix):
    matches = [n for n in zf.namelist() if n.replace("\\", "/").endswith(suffix)]
    if not matches:
        raise FileNotFoundError(f"Could not find '{suffix}' inside {zf.filename}")
    # Prefer the shallowest exact-looking path.
    return sorted(matches, key=lambda x: (x.count("/"), len(x)))[0]


def discover_stage1b_handoff():
    """Locate the Stage-1B handoff produced by the from-scratch refit.

    This stage deliberately reads the STAGE-1B run (the from-scratch refit of
    the locked architecture), not the earlier eight-classifier screening run.
    Directories and archives are both searched so it works whether Stage 1B is
    still on disk in the same runtime or was re-uploaded as a ZIP.
    """
    hits = []
    for root in [BASE_DIR, FALLBACK_DIR]:
        if os.path.isdir(root):
            hits.extend(Path(root).rglob("STAGE1B_HANDOFF.json"))
    hits = sorted(set(hits), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in hits:
        try:
            h = json.loads(Path(p).read_text())
            if h.get("model_name") == LOCKED_MODEL_NAME:
                return json.loads(Path(p).read_text()), str(p), str(Path(p).parent.parent)
        except Exception:
            continue

    # Fall back to a Stage-1B ZIP if the folder is gone.
    zips = []
    for root in [BASE_DIR, FALLBACK_DIR]:
        if os.path.isdir(root):
            zips.extend(Path(root).rglob("Stage1B_*outputs*.zip"))
    zips = sorted(set(zips), key=lambda p: p.stat().st_mtime, reverse=True)
    for z in zips:
        try:
            out = os.path.join(OUTPUT_DIR, "_stage1b_source")
            shutil.rmtree(out, ignore_errors=True); os.makedirs(out, exist_ok=True)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out)
            found = list(Path(out).rglob("STAGE1B_HANDOFF.json"))
            if found:
                h = json.loads(found[0].read_text())
                if h.get("model_name") == LOCKED_MODEL_NAME:
                    return h, str(found[0]), out
        except Exception:
            continue

    raise FileNotFoundError(
        "STAGE1B_HANDOFF.json was not found.\n"
        "Run the Stage-1B cell (the from-scratch refit) first, "
        "or upload its output ZIP to /content.\n"
        "This stage does NOT fall back to the earlier eight-classifier run."
    )


def verify_handoff_csv(handoff):
    """Confirm the CSV on disk is byte-identical to the one Stage 1B used."""
    expected = handoff.get("input_csv_md5")
    candidates = []
    for root in [BASE_DIR, FALLBACK_DIR]:
        if os.path.isdir(root):
            candidates.extend(Path(root).glob("ML_Ready_Matrix_Time*.csv"))
    stated = handoff.get("input_csv")
    if stated and os.path.exists(stated):
        candidates.insert(0, Path(stated))
    candidates = list(dict.fromkeys(candidates))
    checked = []
    for c in candidates:
        try:
            h = md5_file(str(c)); checked.append((str(c), h))
            if h == expected:
                return str(c)
        except Exception:
            pass
    msg = "\n".join(f"  {a}: {b}" for a, b in checked) or "  no candidate CSVs found"
    raise RuntimeError(
        "The exact CSV used by Stage 1B was not found.\n"
        f"Expected MD5: {expected}\nCandidates checked:\n{msg}"
    )


def find_extracted(root, basename):
    hits = list(Path(root).rglob(basename))
    if not hits:
        raise FileNotFoundError(f"Missing {basename} in extracted Stage-1 output.")
    return str(sorted(hits, key=lambda p: (len(p.parts), len(str(p))))[0])


class SSIPreprocessor:
    def __init__(self, continuous_features, binary_features, ordinal_features,
                 nominal_features, preferred_nominal_levels, scale_continuous=False):
        self.continuous_features = list(continuous_features)
        self.binary_features = list(binary_features)
        self.ordinal_features = list(ordinal_features)
        self.nominal_features = list(nominal_features)
        self.preferred_nominal_levels = preferred_nominal_levels
        self.scale_continuous = scale_continuous
        self.continuous_imputer = None
        self.continuous_scaler = None
        self.discrete_imputer = None
        self.onehot = None
        self.output_feature_names_ = []
        self.missing_indicator_features_ = []

    @staticmethod
    def _parse_binary(x):
        sx = norm_text(x)
        if sx is None:
            return np.nan
        if sx in {"1", "1.0", "yes", "y", "true", "t", "present", "positive"}:
            return 1.0
        if sx in {"0", "0.0", "no", "n", "false", "f", "absent", "negative"}:
            return 0.0
        try:
            v = float(sx)
            return float(v) if v in (0.0, 1.0) else np.nan
        except Exception:
            return np.nan

    @staticmethod
    def _parse_ordinal(x, f):
        sx = norm_text(x)
        if sx is None:
            return np.nan
        if f in ORDINAL_MAPS and sx in ORDINAL_MAPS[f]:
            return float(ORDINAL_MAPS[f][sx])
        try:
            return float(min(max(int(round(float(sx))), 0), 3))
        except Exception:
            return np.nan

    @staticmethod
    def _nominal(f, x):
        x = clean_scalar(x)
        if pd.isna(x):
            return np.nan
        if f == "any_iv_redosed_in_cefazolin_opportunity":
            v = SSIPreprocessor._parse_binary(x)
            if v == 1.0:
                return "Redosed"
            if v == 0.0:
                return "Not_redosed"
            return np.nan
        return str(x).strip()

    def _parts(self, X):
        cont = pd.DataFrame(index=X.index)
        disc = pd.DataFrame(index=X.index)
        nom = pd.DataFrame(index=X.index)
        for c in self.continuous_features:
            cont[c] = pd.to_numeric(X[c].map(clean_scalar), errors="coerce")
        for c in self.binary_features:
            disc[c] = X[c].map(self._parse_binary).astype(float)
        for c in self.ordinal_features:
            disc[c] = X[c].map(lambda z: self._parse_ordinal(z, c)).astype(float)
        for c in self.nominal_features:
            nom[c] = X[c].map(lambda z: self._nominal(c, z)).astype("object")
        return cont, disc, nom

    def _fill_nominal(self, nom):
        out = nom.copy()
        for c in self.nominal_features:
            out[c] = out[c].astype("object").where(
                out[c].notna(), NOMINAL_MISSING_LEVEL.get(c, "Missing")
            )
        return out

    def fit(self, X):
        cont, disc, nom = self._parts(X)
        self.continuous_imputer = SimpleImputer(strategy="median")
        self.discrete_imputer = SimpleImputer(strategy="most_frequent")
        if ADD_CONTINUOUS_MISSING_INDICATORS:
            self.missing_indicator_features_ = [
                c for c in self.continuous_features
                if float(cont[c].isna().mean()) >= MISSING_INDICATOR_MIN_RATE
            ]
        cont_imp = self.continuous_imputer.fit_transform(cont)
        if self.scale_continuous:
            self.continuous_scaler = StandardScaler().fit(cont_imp)
        self.discrete_imputer.fit(disc)
        nomi = self._fill_nominal(nom)
        cats = []
        for c in self.nominal_features:
            obs = set(nomi[c].astype(str).unique().tolist())
            pref = [p for p in self.preferred_nominal_levels.get(c, []) if p in obs]
            cats.append(pref + sorted(obs - set(pref)))
        try:
            self.onehot = OneHotEncoder(categories=cats, handle_unknown="ignore", sparse_output=False)
        except TypeError:
            self.onehot = OneHotEncoder(categories=cats, handle_unknown="ignore", sparse=False)
        self.onehot.fit(nomi.astype(str))
        self.output_feature_names_ = (
            self.continuous_features + self.binary_features + self.ordinal_features +
            self.onehot.get_feature_names_out(self.nominal_features).tolist() +
            [f"{c}__was_missing" for c in self.missing_indicator_features_]
        )
        return self

    def transform(self, X):
        cont, disc, nom = self._parts(X)
        ind = (
            cont[self.missing_indicator_features_].isna().astype(float).to_numpy()
            if self.missing_indicator_features_
            else np.zeros((len(X), 0), dtype=float)
        )
        a = self.continuous_imputer.transform(cont)
        if self.scale_continuous and self.continuous_scaler is not None:
            a = self.continuous_scaler.transform(a)
        b = self.discrete_imputer.transform(disc)
        nomi = self._fill_nominal(nom)
        c = self.onehot.transform(nomi.astype(str))
        return np.concatenate([a, b, c, ind], axis=1).astype(float)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def make_weights(y, mult):
    y = np.asarray(y).astype(int)
    npos = int(np.sum(y == 1)); nneg = int(np.sum(y == 0))
    if npos == 0:
        raise ValueError("No positive events in training data.")
    positive_weight = (nneg / max(npos, 1)) * float(mult)
    return np.where(y == 1, positive_weight, 1.0).astype(float)


def build_model(model_name, params, seed):
    """Rebuild the Stage-1B architecture exactly as it was fitted there."""
    p = {k: v for k, v in params.items() if k != "positive_weight_multiplier"}
    if model_name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(objective="binary:logistic", eval_metric="aucpr",
                             random_state=int(seed), n_jobs=-1,
                             tree_method="hist", **p)
    if model_name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(random_state=int(seed), **p)
    raise ValueError(
        f"This SHAP stage supports XGBoost and HistGradientBoosting; got {model_name}."
    )


def fit_hgb(X, y, params, seed, model_name=None):
    """Fit the locked architecture on the given data. Name kept for continuity."""
    pre = SSIPreprocessor(
        CONTINUOUS_FEATURES, BINARY_FEATURES, ORDINAL_FEATURES,
        NOMINAL_FEATURES, PREFERRED_NOMINAL_LEVELS, scale_continuous=False
    )
    Xt = pre.fit_transform(X[ALL_FEATURES])
    model = build_model(model_name or LOCKED_MODEL_NAME, params, seed)
    w = make_weights(y, params.get("positive_weight_multiplier", 1.0))
    model.fit(Xt, np.asarray(y).astype(int), sample_weight=w)
    return pre, model


def predict_raw(pre, model, X):
    Xt = pre.transform(X[ALL_FEATURES])
    return np.asarray(model.predict_proba(Xt))[:, 1].astype(float)


GRADE_LABELS = {0: "Grade 1 - Low risk", 1: "Grade 2 - Intermediate risk",
                2: "Grade 3 - High risk"}


def assign_grade(score, c1, c2):
    score = np.asarray(score, dtype=float)
    g = np.full(len(score), 2, dtype=int)
    g[score < c1] = 0
    g[(score >= c1) & (score < c2)] = 1
    return g


def load_scorer(path="grading_model.joblib"):
    """Load the artefact, making it loadable outside its creating session.

    The artefact was pickled from a script, so the preprocessor class is
    recorded under __main__. Aliasing the class into __main__ before unpickling
    lets the artefact load from any entry point without re-saving it.
    """
    import sys
    main = sys.modules.get("__main__")
    if main is not None:
        for name in ("SSIPreprocessor",):
            if not hasattr(main, name) and name in globals():
                setattr(main, name, globals()[name])
    return joblib.load(path)


def score_patient(record, scorer):
    pre = scorer["preprocessor"]; model = scorer["model"]
    cal = scorer["calibrator"]; cut = scorer["cutoffs"]
    row = {f: record.get(f, np.nan) for f in ALL_FEATURES}
    X = pd.DataFrame([row])
    raw = float(np.asarray(model.predict_proba(pre.transform(X[ALL_FEATURES])))[:, 1][0])
    risk = float(cal.predict([raw])[0]) if cal is not None else raw
    g = int(assign_grade(np.array([risk]), cut["cutoff_grade0_to_1"],
                         cut["cutoff_grade1_to_2"])[0])
    obs = (scorer.get("meta", {}).get("test_rates", {}) or {}).get(g)
    return {"calibrated_risk": risk, "raw_score": raw, "grade": g,
            "grade_label": GRADE_LABELS[g],
            "observed_rate_in_grade_test_split": obs}


def score_dataframe(df, scorer):
    pre = scorer["preprocessor"]; model = scorer["model"]
    cal = scorer["calibrator"]; cut = scorer["cutoffs"]
    X = df.copy()
    for f in ALL_FEATURES:
        if f not in X.columns:
            X[f] = np.nan
    raw = np.asarray(model.predict_proba(pre.transform(X[ALL_FEATURES])))[:, 1]
    risk = cal.predict(raw) if cal is not None else raw
    g = assign_grade(risk, cut["cutoff_grade0_to_1"], cut["cutoff_grade1_to_2"])
    out = df.copy()
    out["calibrated_risk"] = risk
    out["risk_grade"] = g
    out["risk_grade_label"] = [GRADE_LABELS[int(v)] for v in g]
    return out
