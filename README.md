# Paediatric neurosurgery SSI risk grading app

Streamlit interface for the locked HistGradientBoosting SSI risk model.

## Repository contents

All four files must sit in the **same folder** (the repository root is simplest):

```
app.py                  Streamlit interface
ssi_scorer.py           scoring module produced by the analysis
grading_model.joblib    fitted preprocessor, model, calibrator, cut points
requirements.txt        pinned dependencies
```

## Deploying

1. Push the four files to a public GitHub repository.
2. Go to https://share.streamlit.io and choose **New app**.
3. Select the repository and branch.
4. Set **Main file path** to `app.py`.
5. Deploy.

## Why the versions are pinned

`requirements.txt` pins `scikit-learn==1.6.1` and `numpy==2.0.2`. These are not
cosmetic. The model artefact was fitted under those versions, and both bounds
were confirmed by testing:

- scikit-learn 1.8 loads the file but then fails inside `SimpleImputer.transform`
  with `AttributeError: '_fill_dtype'`
- numpy 1.26 fails at load time with `ValueError: PCG64 is not a known
  BitGenerator module`

If either is left unpinned, Streamlit Cloud installs the newest release and the
app breaks after deployment even though it worked locally. Do not relax these
pins without re-exporting the model.

## Antibiotic regimen categories

The interface offers three regimen categories: cefazolin alone, cefazolin plus
vancomycin, and other. The deployed model artefact was fitted with eight
regimen levels, so five of them are not selectable. A patient whose regimen was
cefuroxime, clindamycin, nafcillin or vancomycin alone must therefore be entered
as "Other regimen", which is not the category the model used for them during
fitting, and the prediction for those patients is correspondingly less reliable.
The app states this beneath the dropdown.

To remove the discrepancy, re-run the analysis on the three-level version of the
variable and re-export `grading_model.joblib`. The restricted list is set in
`NOMINAL_UI_OPTIONS` in `app.py`; each entry is checked against the fitted
encoder at start-up, so an option that the model does not recognise is dropped
with a visible warning rather than silently producing an unseen category.

## Cefazolin timing

Missing cefazolin timing is informative in this model, which carries an explicit
missingness indicator for it. The form therefore offers a **"Cefazolin timing not
recorded"** checkbox rather than forcing a number. Use it whenever no timing is
documented; entering an invented value will change the prediction.

## Limitations

Research tool. It reports the model's risk estimate from recorded variables. It
does not establish causation, so changing a modifiable input such as antibiotic
timing is not shown to change risk. It is not a substitute for clinical
judgement.
