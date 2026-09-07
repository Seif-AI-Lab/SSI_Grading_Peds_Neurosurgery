# Paediatric neurosurgery SSI risk grading app

Streamlit interface for the locked SSI risk-grading model.

## Required repository files

Keep these files together in the repository root:

```text
app.py
ssi_scorer.py
grading_model.joblib
requirements.txt
```

## Streamlit Community Cloud

Set the **Main file path** to `app.py`.

This serialized model uses pinned scientific-Python dependencies and should be deployed with **Python 3.12**. If an existing Streamlit deployment was created with Python 3.14, delete it and redeploy; choose **Python 3.12** under **Advanced settings** during deployment.

The filenames `ssi_scorer.py` and `grading_model.joblib` must be preserved exactly.

## Intended use

Research use only. The application reports the locked model's estimated 30-day SSI risk and risk grade and is not a substitute for clinical judgement.
