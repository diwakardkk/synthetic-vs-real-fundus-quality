"""Mixed-effects logistic model and GEE check for the LQ effect on correctness.

``correct ~ lq + C(grade) + (1|patient) + (1|model)`` fitted with statsmodels'
``BinomialBayesMixedGLM`` (variational Bayes); GEE with patient clusters as a frequentist check.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def mixed_logit(df: pd.DataFrame) -> dict:
    """df columns: correct (0/1), lq (0/1), grade, patient_id, model."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    d = df[["correct", "lq", "grade", "patient_id", "model"]].dropna().copy()
    d["grade"] = d["grade"].astype(int).astype(str)
    if d["lq"].nunique() < 2 or len(d) < 50:
        return {"or": np.nan, "ci": [np.nan, np.nan], "note": "insufficient data"}
    vc = {"patient": "0 + C(patient_id)", "model": "0 + C(model)"}
    formula = "correct ~ lq + C(grade)" if d["grade"].nunique() > 1 else "correct ~ lq"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = BinomialBayesMixedGLM.from_formula(formula, vc, d)
        r = m.fit_vb()
    names = list(m.exog_names)
    i = names.index("lq")
    mean, sd = float(r.fe_mean[i]), float(r.fe_sd[i])
    return {"or": float(np.exp(mean)), "ci": [float(np.exp(mean - 1.96 * sd)), float(np.exp(mean + 1.96 * sd))],
            "log_or": mean, "log_or_sd": sd, "n": int(len(d)), "method": "BinomialBayesMixedGLM (VB)"}


def gee_logit(df: pd.DataFrame) -> dict:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    d = df[["correct", "lq", "grade", "patient_id", "model"]].dropna().copy()
    d["grade"] = d["grade"].astype(int).astype(str)
    if d["lq"].nunique() < 2 or len(d) < 50:
        return {"or": np.nan, "ci": [np.nan, np.nan], "p": np.nan, "note": "insufficient data"}
    formula = "correct ~ lq + C(grade) + C(model)" if d["grade"].nunique() > 1 else "correct ~ lq + C(model)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = smf.gee(formula, "patient_id", d, family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
    b, se = float(r.params["lq"]), float(r.bse["lq"])
    return {"or": float(np.exp(b)), "ci": [float(np.exp(b - 1.96 * se)), float(np.exp(b + 1.96 * se))],
            "p": float(r.pvalues["lq"]), "n": int(len(d)), "method": "GEE exchangeable, patient clusters"}
