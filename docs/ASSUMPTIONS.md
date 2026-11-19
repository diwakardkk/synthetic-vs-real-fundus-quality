# Assumptions to verify on the server

Run `python -m realcal.cli inspect --dataset <name>` for each dataset and check the items below.

## Dataset layouts

Items marked **[verified]** were checked against the real files while the code was being written. Items marked **[assumed]** were not.

### DeepDRiD (`configs/datasets/deepdrid.yaml`)
1. **[verified]** Regular-fundus subsets `regular-fundus-training`, `regular-fundus-validation` and `Online-Challenge1&2-Evaluation`. Each has an `Images/<patient>/<image_id>.jpg` tree.
2. **[verified]** The training/validation CSV columns are `patient_id, image_id, image_path, Overall quality, left_eye_DR_Level, right_eye_DR_Level, patient_DR_Level, Clarity, Field definition, Artifact`. A `*-source-*.csv` gives `Source` (used as `site`).
3. **[verified]** Test labels come from `Challenge1_labels.xlsx` (`image_id, DR_Levels`) and `Challenge2_labels.xlsx` (`image_id, Overall quality, Artifact, Clarity, Field definition`). The test set is merged into the pool with `official_split = test`. There is no site information for the test set.
4. **[assumed]** View suffix `1` = macula-centred and `2` = optic-disc-centred (`view_map`). Check this against the DeepDRiD readme. It only affects labelling, not the analyses.
5. **[assumed]** The quality scales and thresholds are: `hq = Overall quality == 1`; `lq_artifact = Artifact >= 6`; `lq_clarity = Clarity <= 4`; `lq_field = Field definition <= 4`. Confirm them from the `inspect` cross-tabulations. Defect flags are only set on non-HQ images, so each defect stratum is a subset of the LQ stratum.
6. The image-level grade is the eye-level DR level for that image's laterality.

### FIVES (`configs/datasets/fives.yaml`)
7. **[verified]** `train/Original/<n>_<A|D|G|N>.png` (600) and `test/Original/...` (200). Numbering restarts in the test set, so image ids are prefixed `train_`/`test_`.
8. **[verified]** `Quality Assessment.xlsx` has sheets `Train`/`Test` with columns `Disease, Number, IC, Blur, LC`, joined on `Number`.
9. **[assumed]** Quality value `1` = good (`good_value: 1`). In the train sheet, 459 of 600 images are `1,1,1`, which supports this. With this assumption the official test set has 67 LQ images, which is above the 30-image minimum, so the CV variant is not triggered by default.

### MSHF (`configs/datasets/mshf.yaml`)
10. **[verified]** Colour fundus images: `Original/CFP/{DR-XJU 235, DR-ZJU 187, Glaucoma 52, Healthy 26}` = 500 tabletop, and `Original/Portable_camera/{Local1 199, Local2 103}` = 302 portable. `UWF-mosaic` and the `AI-use` copies are skipped.
11. **[verified]** The combined "gold standard" sheet (named `DR-ZJU`) in `MSHF_quality_scores.xlsx` contains formulas `=IF(a1+a2+a3>1,1,0)`, a majority vote of 3 annotators. Their saved results are all 0 in the distributed file. The loader therefore recomputes the majority vote from `Individual_scores.xlsx.xlsx` (3 annotators × `illumination, clarity, contrast, Overall`; all 1,302 images). The per-folder sheets are used only for images not in that workbook, and agreement between the two is logged. Ten byte-identical duplicate images in `Portable_camera/Local1` are dropped (the first id is kept).
12. **[verified]** `1` = good (per-folder sheets: about 50% good overall). `hq = Overall == 1`.

### DDR (`configs/datasets/ddr.yaml`)
13. **[assumed]** After joining and unzipping the parts: `DDR-dataset/DR_grading/{train,valid,test}/*.jpg` with `{train,valid,test}.txt` lines `"<file> <grade>"`, where grade 5 = ungradable. The loader searches below `NPJ/DDR` for a `DR_grading` folder.
14. **[assumed]** Patient ids are not used; each image is its own patient. Set `patient_regex` if the file names encode patients. The official `valid` split is unused by default (`use_official_valid_as: none`).

## Weights (`configs/backbones.yaml`)
15. **[assumed]** RETFound: HF `YukunZhou/RETFound_mae_natureCFP`, file `RETFound_mae_natureCFP.pth`, key `model`, timm `vit_large_patch16_224`, global average pooling (the MAE `norm` is loaded into `fc_norm`).
16. **[assumed]** VisionFM: checkpoint key `teacher`, prefixes `module.`/`backbone.` stripped, timm `vit_base_patch16_224`, CLS token. Loading fails clearly if more than 20% of tensors are missing; then adjust `state_key` / `strip_prefixes`.
17. **[assumed]** FLAIR: torchvision ResNet-50 with keys prefixed `vision_model.model.`, ImageNet normalisation.
18. **[assumed]** BiomedCLIP: only the image tower is loaded from `open_clip_pytorch_model.bin` (`visual.trunk.*` → timm `vit_base_patch16_224`, CLS token after the final norm), with the OpenAI CLIP mean/std. This avoids needing the text tower offline.
19. **[assumed]** DINOv3: Transformers snapshot (needs `transformers>=4.56`); `pooler_output` is the CLS embedding; LoRA on `q_proj`/`v_proj`.
20. **[assumed]** timm hub weights: `vit_base_patch14_dinov2.lvd142m`, `resnet50.tv_in1k`, `efficientnet_b3.ra2_in1k`, `convnext_base.fb_in1k`.

## Method choices made where the brief left room
21. The LoRA "query and value" projections in timm ViTs are fused (`qkv`). A custom module adds rank-8 updates to the q and v slices only.
22. For the FIVES 4-class task, sensitivity/specificity and the 90%-specificity threshold use the score `1 - P(normal)`. AUROC is macro one-vs-rest.
23. "Top-5 overlap" uses the five most robust configurations (smallest loss).
24. The two-level bootstrap CIs for LOBO-fitted methods apply the per-fold fitted weights to each bootstrap replicate's losses; they do not refit inside each replicate. The group-weight CIs (Table 5) do refit on each replicate.
25. Level-2 `pi` uses only images (no model losses), so it needs no LOBO.
26. **[verified, design change]** DeepDRiD grades quality per eye: both views of an eye share the score in about 99% of eyes, so within-eye HQ/LQ pairs are about 0. Control 2 therefore uses within-patient contralateral pairs: one all-HQ eye and one all-LQ eye with the same DR grade. That gives 72 patients in training+validation alone. The decision is made at eye level (the eye is referable if any of its images is positive). Set `pair_level: eye` in `deepdrid.yaml` to go back to within-eye pairs. McNemar tests use seed 0. The pooled test is descriptive because the pairs are reused across configurations.
27. `last.pt` is deleted after a training job finishes (only `best.pt` is kept) to save disk space on full fine-tunes.
28. Analysis stages (`exp1`–`exp3`, `clinical`, …) re-run on every call. Their expensive bootstrap results are cached under `cache/analysis/`, keyed by a fingerprint of the prediction store.
