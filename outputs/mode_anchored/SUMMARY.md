# Sleep model comparison

Five stages: Wake, N1, N2, N3, REM. All percentages below are measured on final held-out nights.

Personalization mode: **anchored**. Anchored keeps each target person's validation and test nights fixed while history grows backward; walk_forward trains on nights 1..k, validates night k+1, and tests night k+2, mirroring live device learning.

Population training excludes the target subject's outer fold. Both models use identical sensor features and source/target splits. Source validation chooses model settings and neural-network stopping. Target validation chooses personalization weight and causal smoothing. History k counts training nights; one additional labeled validation night is required. k=0 is a population-only baseline, with no target validation tuning.

Temporal features use current and preceding 1/5/15-minute windows and elapsed time. No future sensor values or true prior stages enter predictions. A compact two-layer MLP is the neural comparator; this is not a reproduction of the published LSTM.

| Cohort | Model | People | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| all_subjects_personalized_k1_plus | majority_stage | 47 | 35.7% | 0.104 |
| all_subjects_personalized_k1_plus | neural_network | 47 | 56.9% | 0.494 |
| all_subjects_personalized_k1_plus | original_rf | 47 | 30.6% | 0.241 |
| all_subjects_personalized_k1_plus | random_forest | 47 | 56.5% | 0.476 |
| all_subjects_max_available_history | majority_stage | 47 | 36.4% | 0.106 |
| all_subjects_max_available_history | neural_network | 47 | 56.6% | 0.488 |
| all_subjects_max_available_history | original_rf | 47 | 31.1% | 0.249 |
| all_subjects_max_available_history | random_forest | 47 | 56.2% | 0.469 |
| common_cohort_k0 | neural_network | 11 | 56.8% | 0.503 |
| common_cohort_k0 | random_forest | 11 | 59.4% | 0.530 |
| common_cohort_k1 | majority_stage | 11 | 32.9% | 0.098 |
| common_cohort_k1 | neural_network | 11 | 58.4% | 0.523 |
| common_cohort_k1 | original_rf | 11 | 32.1% | 0.263 |
| common_cohort_k1 | random_forest | 11 | 59.7% | 0.521 |
| common_cohort_k2 | majority_stage | 11 | 37.0% | 0.108 |
| common_cohort_k2 | neural_network | 11 | 59.5% | 0.528 |
| common_cohort_k2 | original_rf | 11 | 33.3% | 0.276 |
| common_cohort_k2 | random_forest | 11 | 61.6% | 0.541 |
| common_cohort_k3 | majority_stage | 11 | 37.5% | 0.109 |
| common_cohort_k3 | neural_network | 11 | 59.4% | 0.528 |
| common_cohort_k3 | original_rf | 11 | 36.7% | 0.297 |
| common_cohort_k3 | random_forest | 11 | 62.2% | 0.538 |
| common_cohort_k4 | majority_stage | 11 | 37.5% | 0.109 |
| common_cohort_k4 | neural_network | 11 | 59.2% | 0.527 |
| common_cohort_k4 | original_rf | 11 | 36.0% | 0.293 |
| common_cohort_k4 | random_forest | 11 | 61.5% | 0.533 |
| common_cohort_k5 | majority_stage | 11 | 37.5% | 0.109 |
| common_cohort_k5 | neural_network | 11 | 59.1% | 0.521 |
| common_cohort_k5 | original_rf | 11 | 35.5% | 0.295 |
| common_cohort_k5 | random_forest | 11 | 60.8% | 0.517 |

The original RF baseline is retrained on exactly the same target training nights, using its original features/settings; it has no population training. Its contrast with the new models measures the entire pipeline change, not just the classifier. The two new models share all available training data.

The main cohort (all_subjects_personalized_k1_plus) averages every personalized history k>=1, giving each person equal weight: per-subject means first, then the subject mean. This cohort distinguishes anchored from walk_forward evaluation; the max-history cohort is identical in both because splits coincide at k=max. Only the common cohort supports a controlled 0-5-night curve. Participants, not epochs, are the statistical units. Test results must not be used to retune this experiment. No clinical reliability threshold or minimum-night claim is established.
