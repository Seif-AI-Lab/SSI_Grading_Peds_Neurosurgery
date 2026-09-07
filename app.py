# -*- coding: utf-8 -*-
"""
Streamlit application for the paediatric neurosurgery SSI risk grading system.

Repository layout expected by Streamlit Community Cloud:

    app.py                  this file
    ssi_scorer.py           scoring module produced by the analysis
    grading_model.joblib    fitted preprocessor, model, calibrator, cut points
    requirements.txt        pinned dependencies

Main file path in the Streamlit deployment dialog: app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

from ssi_scorer import (
    load_scorer, score_patient,
    ALL_FEATURES, CONTINUOUS_FEATURES, BINARY_FEATURES,
    ORDINAL_FEATURES, NOMINAL_FEATURES, FEATURE_LABELS,
)

st.set_page_config(page_title="Paediatric neurosurgery SSI risk grade",
                   page_icon="🧠", layout="wide")

MODEL_FILE = "grading_model.joblib"

GRADE_STYLE = {
    0: {"name": "LOW RISK",          "bg": "#E8F5E9", "fg": "#1B5E20", "bar": "#2E7D32"},
    1: {"name": "INTERMEDIATE RISK", "bg": "#FFF4E5", "fg": "#8A4B00", "bar": "#EF6C00"},
    2: {"name": "HIGH RISK",         "bg": "#FDECEA", "fg": "#8E1B14", "bar": "#C62828"},
}

# Presentation labels for the 19 predictors. These affect the interface only;
# the underlying variable names passed to the model are unchanged.
PRETTY_LABELS = {
    "age_years": "Age at operation (years)",
    "cefazolin_time_min": "Cefazolin administration relative to incision (minutes)",
    "OPTIME": "Total operative time (minutes)",
    "sex_binary": "Male sex",
    "ASA_ge3": "ASA physical status III or higher",
    "CASETYPE": "Emergency or urgent case",
    "ACQ_ABNORMALITY": "Acquired central nervous system abnormality",
    "IMPCOGSTAT": "Impaired cognitive status",
    "SEIZURE": "Seizure disorder",
    "premature_birth_any": "History of premature birth",
    "NEUROMUSCDIS": "Neuromuscular disorder",
    "CEREBRAL_PALSY": "Cerebral palsy",
    "NUTR_SUPPORT": "Preoperative nutritional support",
    "cardiac_comorbidity_any": "Cardiac comorbidity",
    "STRUCT_PULM_AB": "Structural pulmonary abnormality",
    "additional_procedure_burden": "Number of additional concurrent procedures",
    "procedure_group": "Procedure category",
    "regimen_group": "Antibiotic prophylaxis regimen",
    "any_iv_redosed_in_cefazolin_opportunity": "Intraoperative antibiotic redosing",
}

# Presentation labels for categorical levels.
LEVEL_LABELS = {
    "Not_redosed": "Not redosed",
    "Redosed": "Redosed",
    "No_redosing_opportunity": "No redosing opportunity",
    "Cefazolin alone": "Cefazolin alone",
    "Cefazolin + vancomycin": "Cefazolin plus vancomycin",
    "Other": "Other regimen",
}

# Levels offered in the interface, where these are deliberately narrower than
# the levels the model was fitted with. Any entry here is checked against the
# fitted encoder at start-up, so a typo cannot silently produce an all-zero
# encoding that the model would score as an unseen category.
NOMINAL_UI_OPTIONS = {
    "regimen_group": ["Cefazolin alone", "Cefazolin + vancomycin", "Other"],
}

# Sensible entry ranges, taken from the development cohort.
CONT_SPEC = {
    "age_years":          {"min": 0.0,    "max": 18.0,   "default": 6.2,   "step": 0.1,
                           "help": "Age at operation in years (0 to 18)."},
    "cefazolin_time_min": {"min": -240.0, "max": 60.0,   "default": -18.0, "step": 1.0,
                           "help": "Minutes relative to incision. Negative means the dose "
                                   "was given before incision."},
    "OPTIME":             {"min": 1.0,    "max": 1200.0, "default": 91.0,  "step": 1.0,
                           "help": "Total operative time in minutes."},
}


@st.cache_resource(show_spinner="Loading the locked model ...")
def get_scorer():
    if not os.path.exists(MODEL_FILE):
        return None, (f"{MODEL_FILE} was not found. Upload it to the same folder as "
                      f"app.py in the repository.")
    try:
        return load_scorer(MODEL_FILE), None
    except Exception as exc:
        return None, (f"The model file could not be loaded ({type(exc).__name__}: {exc}). "
                      f"This is usually a library-version mismatch: the artefact was "
                      f"created with scikit-learn 1.6.1, so requirements.txt must pin "
                      f"that exact version.")


def model_levels(scorer, feature):
    """Levels the fitted one-hot encoder actually recognises for this variable."""
    pre = scorer["preprocessor"]
    try:
        i = pre.nominal_features.index(feature)
        return [str(v) for v in pre.onehot.categories_[i]]
    except Exception:
        return []


def nominal_options(scorer, feature):
    """Levels to offer in the interface.

    Defaults to every level the encoder knows. Where NOMINAL_UI_OPTIONS narrows
    a variable, only those levels are offered, and each is verified to exist in
    the fitted encoder so the value sent to the model is always one it was
    trained on.
    """
    known = model_levels(scorer, feature)
    wanted = NOMINAL_UI_OPTIONS.get(feature)
    if not wanted:
        return known, []
    valid = [v for v in wanted if v in known]
    unknown = [v for v in wanted if v not in known]
    return (valid or known), unknown


def label_for(feature):
    return PRETTY_LABELS.get(feature,
                             FEATURE_LABELS.get(feature, feature.replace("_", " ")))


scorer, load_error = get_scorer()

st.title("Paediatric neurosurgery SSI risk grade")
st.caption("Thirty-day surgical site infection risk from 19 preoperative and "
           "intraoperative variables, reported as a three-level risk grade.")

if load_error:
    st.error(load_error)
    st.stop()

cut = scorer["cutoffs"]
meta = scorer.get("meta", {}) or {}
test_rates = meta.get("test_rates", {}) or {}

with st.sidebar:
    st.header("About this tool")
    st.markdown(
        f"""
**Grade boundaries** (calibrated risk)

- Low: below {cut['cutoff_grade0_to_1']:.4f}
- Intermediate: {cut['cutoff_grade0_to_1']:.4f} to {cut['cutoff_grade1_to_2']:.4f}
- High: {cut['cutoff_grade1_to_2']:.4f} and above

Boundaries are the one-third and two-thirds quantiles of calibrated risk in the
calibration set. No held-out test data was used to place them.
"""
    )
    if test_rates:
        st.markdown("**Observed 30-day SSI rate per grade in the held-out test set**")
        st.markdown("\n".join(
            f"- {GRADE_STYLE[int(g)]['name'].title()}: {float(r) * 100:.2f}%"
            for g, r in sorted(test_rates.items(), key=lambda kv: int(kv[0]))))
    st.divider()
    st.warning("Research tool. It reports the model's risk estimate from recorded "
               "variables, does not establish causation, and is not a substitute "
               "for clinical judgement.")

st.subheader("Patient and operative details")

record = {}

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Continuous variables**")
    for f in CONTINUOUS_FEATURES:
        spec = CONT_SPEC.get(f, {"min": 0.0, "max": 1000.0, "default": 0.0, "step": 1.0,
                                 "help": ""})
        if f == "cefazolin_time_min":
            # Missing cefazolin timing is informative in this model: the training
            # pipeline carries an explicit missingness indicator for it. The form
            # therefore allows "not recorded" rather than forcing a number.
            not_recorded = st.checkbox("Cefazolin timing not recorded", value=False,
                                       key="cef_missing",
                                       help="Tick when no cefazolin timing is "
                                            "documented. The model handles this "
                                            "explicitly rather than assuming a value.")
            if not_recorded:
                record[f] = np.nan
                st.caption("Cefazolin timing will be treated as not recorded.")
                continue
        record[f] = st.number_input(
            label_for(f), min_value=float(spec["min"]), max_value=float(spec["max"]),
            value=float(spec["default"]), step=float(spec["step"]),
            help=spec.get("help", ""), key=f"num_{f}")

    st.markdown("**Procedure and antibiotic regimen**")
    for f in ORDINAL_FEATURES:
        record[f] = st.selectbox(label_for(f), ["0", "1", "2", "3+"], index=0,
                                 key=f"ord_{f}",
                                 help="Number of additional procedures performed.")
    for f in NOMINAL_FEATURES:
        opts, unknown = nominal_options(scorer, f)
        if not opts:
            record[f] = np.nan
            continue
        if unknown:
            st.warning(f"These configured options are not levels the model was "
                       f"fitted with and were dropped: {', '.join(unknown)}")
        shown = [LEVEL_LABELS.get(o, o) for o in opts]
        pick = st.selectbox(label_for(f), shown, index=0, key=f"nom_{f}")
        record[f] = opts[shown.index(pick)]
        hidden = [l for l in model_levels(scorer, f) if l not in opts]
        if hidden:
            st.caption(
                f"Only these {len(opts)} categories are offered. The model was "
                f"fitted with {len(opts) + len(hidden)}, so a patient whose "
                f"regimen was {', '.join(hidden[:3])}"
                + (" and others" if len(hidden) > 3 else "")
                + " must be entered as the closest available option, which is "
                  "not how the model represented them during fitting.")

with c2:
    st.markdown("**Comorbidities and status**")
    for f in BINARY_FEATURES:
        record[f] = 1 if st.checkbox(label_for(f), value=False,
                                     key=f"bin_{f}") else 0

st.divider()

if st.button("Calculate risk grade", type="primary", use_container_width=True):
    missing = [f for f in ALL_FEATURES if f not in record]
    for f in missing:
        record[f] = np.nan

    try:
        result = score_patient(record, scorer)
    except Exception as exc:
        st.error(f"Scoring failed ({type(exc).__name__}: {exc}).")
        st.stop()

    g = int(result["grade"])
    style = GRADE_STYLE[g]
    risk = float(result["calibrated_risk"])
    obs = result.get("observed_rate_in_grade_test_split")

    st.markdown(
        f"""
<div style="background:{style['bg']};border-left:12px solid {style['bar']};
            border-radius:10px;padding:22px 26px;margin-top:6px">
  <div style="font-size:13px;letter-spacing:.14em;color:{style['fg']};
              font-weight:700">PREDICTED RISK GRADE</div>
  <div style="font-size:40px;font-weight:800;color:{style['fg']};
              line-height:1.15;margin:6px 0 2px 0">{style['name']}</div>
  <div style="font-size:17px;color:{style['fg']}">
    Estimated 30-day SSI risk: <b>{risk * 100:.2f}%</b>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated risk", f"{risk * 100:.2f}%")
    if obs is not None and obs == obs:
        m2.metric("Observed rate in this grade", f"{float(obs) * 100:.2f}%",
                  help="Observed 30-day SSI rate among held-out test patients "
                       "assigned to this grade.")
    cohort = meta.get("cohort_rate")
    if cohort:
        m3.metric("Cohort rate", f"{float(cohort) * 100:.2f}%",
                  delta=f"{(risk - float(cohort)) * 100:+.2f} pp",
                  help="Overall 30-day SSI rate in the development cohort.")

    # Where this patient sits relative to the two fixed grade boundaries.
    b1 = float(cut["cutoff_grade0_to_1"]) * 100
    b2 = float(cut["cutoff_grade1_to_2"]) * 100
    axis_max = max(risk * 100 * 1.35, b2 * 2.2, 6.0)
    pos = min(risk * 100 / axis_max * 100, 99.0)
    st.markdown(
        f"""
<div style="margin-top:18px">
  <div style="position:relative;height:26px;border-radius:13px;overflow:hidden;
              background:linear-gradient(to right,
              {GRADE_STYLE[0]['bar']} 0%, {GRADE_STYLE[0]['bar']} {b1/axis_max*100:.2f}%,
              {GRADE_STYLE[1]['bar']} {b1/axis_max*100:.2f}%, {GRADE_STYLE[1]['bar']} {b2/axis_max*100:.2f}%,
              {GRADE_STYLE[2]['bar']} {b2/axis_max*100:.2f}%, {GRADE_STYLE[2]['bar']} 100%)">
    <div style="position:absolute;left:{pos:.2f}%;top:-4px;width:4px;height:34px;
                background:#111"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:12px;
              color:#555;margin-top:5px">
    <span>0%</span><span>boundaries at {b1:.2f}% and {b2:.2f}%</span>
    <span>{axis_max:.1f}%</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.expander("Values submitted"):
        # The submitted values mix numbers, category labels and missing entries,
        # so the column is rendered as text: a mixed-type column cannot be
        # converted to Arrow and would raise a display error.
        def show(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "not recorded"
            if isinstance(v, float):
                return f"{v:g}"
            return str(v)
        st.dataframe(
            pd.DataFrame({"variable": [label_for(f) for f in ALL_FEATURES],
                          "value": [show(record.get(f)) for f in ALL_FEATURES]}),
            use_container_width=True, hide_index=True)

    st.caption("Research tool. Not a substitute for clinical judgement.")
else:
    st.info("Enter the patient's details above, then press "
            "**Calculate risk grade**.")
