# Deviations from the stage-1 protocol, and provenance of the freeze

The stage-1 protocol was written before analysis but was **not deposited publicly**. All deviations are listed
below, with when they were made relative to the freeze of the RealCal weights (the stage-2 lock), which precedes
every analysis on external data (FIVES, MSHF, DDR).

Check every row against the internal stage-1 document before submission. Where the "Planned" column says
"(check)", confirm the original wording.

## Timeline

| Date (2026) | Event |
|---|---|
| (protocol date) | Stage-1 protocol written (internal) |
| 18 Nov | First full DeepDRiD development run (21 configurations) |
| 18-19 Nov | Problems found in the development results; fixes D1-D7 made; development rerun (24 configurations) |
| 19 Nov, ~10:53 | `freeze`: stage-2 lock written (SHA-256 of `prereg/frozen_weights.json` and a timestamp) |
| 19 Nov, after lock | External analyses: FIVES, MSHF, DDR, DDR replication; 448 px and augmentation sensitivity |
| 20 Nov | Post-hoc analyses A-F and new figures (no change to any frozen quantity) |

The exact times and hashes are in `results/extras/extras.json` → `E_provenance`. The pipeline refuses every
external-data stage unless the lock exists and matches the hash of the frozen weights (`realcal/prereg.py`,
`require_lock`).

## Deviations

| # | Planned | Done | Why | When | Effect on conclusions |
|---|---|---|---|---|---|
| D1 | FLAIR preprocessing (check) | FLAIR input normalised as /255 with mean 0 and SD 1, matching FLAIR's own preprocessing; FLAIR retrained | The ImageNet mean/SD used first did not match the pretrained encoder, giving implausibly low linear-probe scores | Before lock | FLAIR moves from weakest to among the strongest backbones; main conclusions unchanged |
| D2 | 8 backbones including RETFound | RETFound added late (gated weights were not available for the first run) | Access granted later | Before lock | Configurations go from 21 to 24, as planned |
| D3 | RealCal Level 1a: non-negative L1 fit on raw features, lambda chosen by nested LOBO | Unchanged, plus a pre-lock check: when every LOBO fold is intercept-only, the ranking metrics are reported as not available | The first run selected the largest lambda (intercept-only), so its LOBO rho was an artefact of predicting the mean | Before lock | Prevents reporting an artefactual rho; in the final run, 0 of 8 folds are intercept-only |
| D4 | — | Added a secondary variant, `RealCal_L1a_std` (standardised features, relative lambda grid) | Raw-scale penalty was not comparable across groups | Before lock | Secondary only; reported alongside, not replacing, the primary method |
| D5 | Model selection for deployment (check) | Selection by the highest predicted LQ AUROC = HQ AUROC − predicted loss | Selecting by the smallest predicted loss rewards weak but stable models | Before lock | All rules pick the same model (VisionFM-LoRA) |
| D6 | — | Added the noise ceiling (patient split-half and seed reliability) | To judge how high any predictor's rho could be | Before lock (DeepDRiD); with external analyses (FIVES) | Interpretation only |
| D7 | — | Added case-mix standardisation of the real loss (DeepDRiD by DR grade, FIVES by class) | Quality is confounded with disease severity and class in the dataset tables | Before lock (DeepDRiD); with external analyses (FIVES) | Real loss on DeepDRiD roughly doubles (0.022 → 0.042); FIVES stays negative |
| D8 | MSHF detector evaluation on all sites | Added a sensitivity analysis excluding the Glaucoma and Healthy folders | Those folders are almost entirely LQ or entirely HQ, so detection could exploit the source | Before lock (code); run after lock | Conclusion unchanged (no difference between detectors) |
| D9 | Framing: RealCal as the proposed method | Evaluation framing: synthetic benchmarks vs real failure; RealCal reported as a pre-specified, negative secondary method | Internal and external results do not support RealCal | After external results | Framing only; all pre-specified endpoints are reported in full |
| D10 | — | Post-hoc analyses A-F: pooled validity, precision bounds, sample-size projection, explanation of RealCal's failure, freeze provenance, deferral simulation | To address precision and clinical relevance | After lock | Labelled post hoc in the paper |

Every pre-specified endpoint (H1 zone, primary external endpoint ρ(RealCal) − ρ(B1) on FIVES with Holm
correction, H3 detectors on MSHF and DDR) is reported as planned, whatever the result.

## Zenodo deposit (before submission)

Deposit one record and cite its DOI in the paper's Code availability and Methods sections:

1. `realcal_code/`: the full code, at the state used for the final run.
2. `prereg/frozen_weights.json` and `prereg/stage2_lock.json` from the server, unchanged.
3. The stage-1 protocol document, as written, with its original file date. If it was ever emailed or shared,
   the email can serve as date evidence.
4. `logs/full4.log`, `logs/ext.log` and `logs/final.log`.
5. `results/` and `tables/`, including `results/extras/`.
6. This file.

Server command to bundle items 2, 4 and 5:

```bash
cd /path/to/data/realcal_runs && tar czf ~/realcal_zenodo.tgz prereg logs results tables && ls -lh ~/realcal_zenodo.tgz
```
