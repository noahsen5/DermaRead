# Product Backlog (top items)
1. Data acquisition plan (sources, licenses, skin-tone metadata & consent).
2. Dataset loader + stratified split (patient-level).
3. Train baseline (ResNet18), log metrics (acc/F1), save weights.
4. Add Grad-CAM heatmaps for explanation.
5. Bias check: performance by skin-tone group (if metadata available).
6. Robust eval: bootstrapped CIs; external holdout if possible.
7. UI: show prediction + heatmap + disclaimer + references.
8. Add unit tests (preproc, inference path).
9. Add model card (intended use, limitations).
10. Package app for demo video & poster screenshots.
