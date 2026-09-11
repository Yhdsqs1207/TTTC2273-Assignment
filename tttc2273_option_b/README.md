# TTTC2273 Option B - Coagulant Dosing at a Water Treatment Plant

Complete reproducible coursework package for the selected Option B scenario.

## Reproduce the results

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python code/run_all.py
```

The run regenerates the synthetic dataset, 200-candidate conflict plot, 30-seed benchmark, Wilcoxon tests, a high-budget NSGA-II reference front, the selected knee point, membership plots, and the shipped rule base.

## Fixed protocol

- Dataset: 420 synthetic samples
- Data seed: 20260909
- Gaussian output noise: SD 2.2 mg/L
- Split: 70% train / 30% test for each of 30 seeds
- Search budget: 24 individuals x (initial + 12 generations) = 312 evaluations for random search, scalarised GA, and NSGA-II
- Proposed method: zero-order TSK fuzzy system + from-scratch NSGA-II
- Mandatory 9 Turbidity x pH rules guarantee semantic coverage; 27 interaction/correction rules can be pruned
- Objectives: RMSE and an explicit interpretability cost dominated by active rule count plus small semantic-drift penalties

## Authors

- **XU HAOWEI** (member 1): data generation, TSK model, NSGA-II optimiser, experiment harness, result validation, portfolio page and report; narrates the decision, formulation and results sections.
- **XU HAOWEI**: fuzzy-system section and its reflection.
- **BAO YUHANG**: hybrid-optimiser and recommendation sections and their reflections.

## Deliverables

- `index.html` - portfolio page (video link goes in section 8)
- `report.docx` / `report.pdf` - 6-page report
- `code/` - `run_all.py` reproduces every result, figure and CSV in `results/` and `images/`
- `video_script.md` - 10-12 minute narration script
