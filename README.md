# Apple Watch Sleep Stage Prediction

Personalized, five-stage sleep staging (Wake, N1, N2, N3, REM) from Apple Watch heart-rate and wrist accelerometry, using population pretraining followed by per-person personalization. The central research question is how many nights of an individual's historical data are needed for reliable personalized predictions.

## Abstract

Investigating Historical Data Requirements for Personalized Sleep Prediction Using Wearable Sensors

Wearable devices such as smartwatches provide an accessible way to collect physiological signals associated with sleep, including heart rate measured using photoplethysmography (PPG) and wrist-based accelerometry. Monitoring these signals can enable personalized sleep assessment using machine learning. However, the effectiveness of predictive models can depend on the amount of historical data available, which may not be readily accessible, delaying useful predictions and increasing the burden of data collection. Understanding the relationship between historical wearable data and prediction performance is therefore important for developing efficient personalized sleep monitoring approaches.

This project evaluates how personalized sleep-stage prediction changes as the amount of prior wearable data varies. It uses the publicly available BIDSleep dataset, with multi-night Apple Watch heart rate and accelerometry from 47 subjects and sleep-stage annotations from a Dreem 2 EEG device. Population-trained Random Forest and neural-network models are adapted using each person's prior labeled nights and evaluated using anchored and walk-forward protocols.

The retained experiments achieve approximately 57% five-stage accuracy. They do not establish a reliable minimum number of historical nights or a clinical reliability threshold. More nights per person and a predefined reliability target are needed to assess those questions.

AI Use Disclosure: ChatGPT was used for editing and refining abstract text. All research findings and conclusions are the work of the authors.

## Repository layout

```text
scripts/                              pipeline code and protocol tests
  preprocess_data.py                  raw BIDSleep -> 30-second epoch features
  fetch_sleep_features.py             checksum-verified download + preprocessing
  compare_sleep_models.py             population training + personalization
  plot_training_curves.py             training and test-performance plots
  plot_sleep_stage_timelines.py       sleep-stage timelines from labels
  test_compare_sleep_models.py        causality, split, and leakage checks
outputs/                              curated results from retained experiments
  mode_anchored/                      anchored-mode results
  mode_walk_forward/                  walk-forward-mode results
SleepData/                            dataset license, README, and checksums
requirements.txt                      Python dependencies
```

Raw downloads, processed CSVs, per-epoch predictions, training-search logs, and batch-loss plots are generated locally and ignored by Git.

## Data

[BIDSleep](https://physionet.org/content/bidsleep-dataset/1.0.0/) (Song, 2026; DOI 10.13026/a0sy-7t69): multi-night Apple Watch recordings from 47 subjects (3-7 nights each). Each night provides instantaneous heart rate (PPG), three-axis accelerometry, and expert sleep-stage annotations from a Dreem 2 EEG headband. Each 30-second epoch is labeled Wake / N1 / N2 / N3 / REM; unknown epochs (label 5) are excluded from supervised targets.

## Approach

1. **Preprocessing** (`preprocess_data.py`): sensors are aligned to the recording start and aggregated into 30-second epochs. Features include HR summaries (mean/std/min/max/range/count/change from previous epoch) and gravity-adjusted movement (ENMO mean/std/min/max/range).

2. **Temporal features** (`compare_sleep_models.py:temporal_features`): 54 features per epoch - base summaries plus causal 1/5/15-minute rolling context, within-night relative heart rate, missingness indicators, and elapsed time. Features never use future observations and never cross nights.

3. **Population training**: 5-fold subject-level split. For each fold, a Random Forest (200 trees) and a compact two-layer MLP are trained on ~37 source subjects (their final night held out to select model settings and MLP stopping). Target subjects are fully excluded from their fold's population training.

4. **Personalization**: for each target subject, the population models are adapted using k = 1-5 prior labeled training nights, with one additional labeled night reserved for validation. The RF is retrained on the person's nights and blended with population probabilities; the MLP is fine-tuned conservatively from the population weights. The blend weight and temporal smoothing strength are selected on that person's validation night only. k=0 is the population-only baseline and does not use target validation tuning.

5. **Two evaluation modes** (`--mode`):
   - **anchored**: validation and test nights are fixed; only the amount of history varies. Cleanly isolates the effect of history size.
   - **walk_forward**: train on nights 1..k, validate night k+1, test night k+2. Mirrors how a real device improves as it accumulates nights.

## Results

Held-out test performance for five-stage classification across all personalized histories k>=1. Each person contributes equally: results are averaged over histories within each subject, then over the 47 subjects. Anchored evaluation uses a fixed final test night; walk-forward evaluation uses the next test night at each history size.

| Model | Anchored accuracy | Walk-forward accuracy | Anchored macro-F1 | Walk-forward macro-F1 |
|---|---:|---:|---:|---:|
| Random Forest | 56.5% | 57.2% | 0.476 | 0.482 |
| Neural network | 56.9% | 57.2% | 0.494 | 0.498 |
| Personal-only RF baseline | 30.6% | 30.2% | 0.241 | 0.237 |
| Majority-stage baseline | 35.7% | 36.8% | 0.104 | 0.106 |

These values come from the `all_subjects_personalized_k1_plus` cohort in the retained summaries. The population-trained pipelines improve accuracy by about 26-27 percentage points over the personal-only RF baseline; this comparison measures the entire pipeline change, not only the classifier. The common cohort of 11 subjects supports the controlled 0-5-night comparison, which does not establish a reliable minimum-night plateau.

Full results, confidence intervals, and paired neural-network-minus-RF differences:

- [Anchored summary](outputs/mode_anchored/SUMMARY.md) and [detailed results](outputs/mode_anchored/summary.json)
- [Walk-forward summary](outputs/mode_walk_forward/SUMMARY.md) and [detailed results](outputs/mode_walk_forward/summary.json)

## Running

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# From local raw BIDSleep data
python scripts/preprocess_data.py --dataset-path SleepData/a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0 --output processed_sleep_data_enhanced_hr.csv

# Or download the public dataset (checksum-verified, resumable; requires curl)
python scripts/fetch_sleep_features.py --workers 4

# Main experiments (runtime depends on hardware)
python scripts/compare_sleep_models.py --mode anchored --output outputs/mode_anchored
python scripts/compare_sleep_models.py --mode walk_forward --output outputs/mode_walk_forward

# Regenerate plots after running an experiment: validation_search.json is local-only
python scripts/plot_training_curves.py outputs/mode_anchored

# Protocol tests
python -m unittest discover -s scripts -p "test_*.py"
```

## Citation

- Dataset: Song (2026), BIDSleep, PhysioNet, DOI 10.13026/a0sy-7t69
- Source model and publication: [BIDSLabUMass/SLAMSS-IFS](https://github.com/BIDSLabUMass/SLAMSS-IFS), DOI 10.1109/TBME.2025.3612158

## Licensing

The BIDSleep dataset license is retained in [SleepData](SleepData/a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0/LICENSE.txt). It applies to the dataset; a separate license for the project code has not yet been selected.
