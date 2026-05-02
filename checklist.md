## Malicious URL Detection Checklist (CNN-LSTM)

- [ ] 1. Set up Python environment with GPU support and install dependencies: torch, torchvision, scikit-learn, xgboost, lightgbm, pandas, numpy, shap, tensorboard, tqdm.
- [ ] 2. Define a consistent project folder structure (data/raw, data/processed, models, logs, reports).
- [ ] 3. Load URL dataset from `d:\! secURLity\dataset\urls_synthetic_10m.csv` and verify columns: `URL`, `label` (1 malicious, 0 benign).
- [ ] 4. Clean URL data: drop missing rows, strip whitespace, lowercase all URLs.
- [ ] 5. Create stratified 80/10/10 train/val/test splits for URL data and save split files.
- [ ] 6. Build URL character vocabulary from training split and implement fixed-length encoding (pad/truncate to 200 chars).
- [ ] 7. Implement CNN-LSTM model for URL classification (1D-CNN over char embeddings -> LSTM -> classifier head).
- [ ] 8. Train CNN-LSTM with early stopping on validation F1; log metrics to TensorBoard.
- [ ] 9. Evaluate CNN-LSTM on test set and report Accuracy, Precision, Recall, F1.
- [ ] 10. Save CNN-LSTM model weights to `.pt`.

## Malicious URL Detection Checklist (XGBoost)

### Feature Engineering
- [ ] 11. Extract URL structural features from the training split:
  - URL length, hostname length, path length, query string length
  - Path depth (number of `/` segments)
  - Number of query parameters
  - Digit count and digit density (digits / total chars)
  - Special character counts: `-`, `_`, `.`, `@`, `?`, `=`, `&`, `%`
  - Number of subdomains
- [ ] 12. Extract URL scheme and host features:
  - Binary flag: scheme is `http` vs `https`
  - Binary flag: host is an IP address (IPv4/IPv6)
  - Port number (numeric; 0 if absent)
  - TLD extracted via `tldextract`; one-hot or target-encode top-50 TLDs, remainder as `other`
- [ ] 13. Extract lexical heuristic features:
  - Binary flag: URL contains suspicious keywords (`login`, `signin`, `verify`, `secure`, `update`, `restore`, `confirm`, `urgent`, `alert`, `unlock`, `warning`)
  - Binary flag: URL contains scam-bait phrases (`you-won`, `lucky-winner`, `final-notice`, `congratulations`, `limited-time`, `restore-access`, `update-payment`)
  - Binary flag: URL contains a known brand name (`google`, `facebook`, `microsoft`, `instagram`, `twitter`, `amazon`, `apple`, `netflix`, `github`, `azure`, `aws`)
  - Binary flag: path matches known C2 patterns (`/api`, `/report`, `/status`, `/data`, `/admin`, `/config`, `/sync`, `/bin.sh`, `/plugin`, `/i`)
  - File extension category: `cdn` (.js, .css, .jpg, .png, .svg, .gif, .jpeg, .webp, .woff2, .webm, .mp3), `malware` (.exe, .sh, .zip, .scr, .dmg, .dll, .vbs, .ps1, .jar), `script` (.php), `none`
- [ ] 14. Compute entropy of the full URL string (Shannon entropy) and entropy of hostname only.
- [ ] 15. Compute longest consecutive digit run and longest consonant run in hostname.
- [ ] 16. Assemble all features into a flat numeric feature vector; verify no NaN/Inf values remain; save feature column names to `data/processed/xgb_feature_cols.json`.
- [ ] 17. Apply the same feature extraction pipeline to val and test splits using only statistics derived from the training split (no leakage).

### Training
- [ ] 18. Compute class weight ratio (neg/pos) from training split; set `scale_pos_weight` in XGBoost accordingly.
- [ ] 19. Train XGBoost classifier with early stopping on validation F1 (maximize); tune core hyperparameters: `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`.
- [ ] 20. Evaluate XGBoost on test set and report Accuracy, Precision, Recall, F1, ROC-AUC.
- [ ] 21. Save XGBoost model to `models/xgb_url.ubj` and feature column list to `data/processed/xgb_feature_cols.json`.

## Inference & Risk Scoring

- [ ] 22. Implement inference scripts to output two independent probability scores: `score_cnn_lstm` (CNN-LSTM) and `score_xgb` (XGBoost).
- [ ] 23. Map each model's score to risk levels using shared thresholds: 0–25 Low Risk, 26–50 Caution, 51–75 Suspicious, 76–100 High Risk.

## Explainability (SHAP)

- [ ] 24. Run SHAP (DeepExplainer/GradientExplainer) on 200 test samples for CNN-LSTM; generate character-level attribution plots.
- [ ] 25. Run SHAP TreeExplainer on 200 test samples for XGBoost; generate feature importance bar plots and beeswarm summary plots.
- [ ] 26. Save all SHAP outputs and summary figures under `reports/`.