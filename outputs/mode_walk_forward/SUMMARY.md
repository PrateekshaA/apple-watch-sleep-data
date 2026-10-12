# Sleep model comparison

Five stages: Wake, N1, N2, N3, REM. All percentages below are measured on final held-out nights.

Personalization mode: **walk_forward**. Anchored keeps each target person's validation and test nights fixed while history grows backward; walk_forward trains on nights 1..k, validates night k+1, and tests night k+2, mirroring live device learning.

Population training excludes the target subject's outer fold. Both models use identical sensor features and source/target splits. Source validation chooses model settings and neural-network stopping. Target validation chooses personalization weight and causal smoothing. History k counts training nights; one additional labeled validation night is required. k=0 is a population-only baseline, with no target validation tuning.

Temporal features use current and preceding 1/5/15-minute windows and elapsed time. No future sensor values or true prior stages enter predictions. A compact two-layer MLP is the neural comparator; this is not a reproduction of the published LSTM.

| Cohort | Model | People | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| all_subjects_personalized_k1_plus | majority_stage | 47 | 36.8% | 0.106 |
| all_subjects_personalized_k1_plus | neural_network | 47 | 57.2% | 0.498 |
| all_subjects_personalized_k1_plus | original_rf | 47 | 30.2% | 0.237 |
| all_subjects_personalized_k1_plus | random_forest | 47 | 57.2% | 0.482 |
| all_subjects_max_available_history | majority_stage | 47 | 36.4% | 0.106 |
| all_subjects_max_available_history | neural_network | 47 | 56.6% | 0.488 |
| all_subjects_max_available_history | original_rf | 47 | 31.1% | 0.249 |
| all_subjects_max_available_history | random_forest | 47 | 56.2% | 0.469 |
| common_cohort_k0 | neural_network | 11 | 57.8% | 0.528 |
| common_cohort_k0 | random_forest | 11 | 58.9% | 0.538 |
| common_cohort_k1 | majority_stage | 11 | 38.0% | 0.109 |
| common_cohort_k1 | neural_network | 11 | 62.7% | 0.553 |
| common_cohort_k1 | original_rf | 11 | 28.4% | 0.228 |
| common_cohort_k1 | random_forest | 11 | 58.7% | 0.512 |
| common_cohort_k2 | majority_stage | 11 | 35.6% | 0.105 |
| common_cohort_k2 | neural_network | 11 | 55.7% | 0.490 |
| common_cohort_k2 | original_rf | 11 | 30.3% | 0.250 |
| common_cohort_k2 | random_forest | 11 | 53.6% | 0.464 |
| common_cohort_k3 | majority_stage | 11 | 34.8% | 0.103 |
| common_cohort_k3 | neural_network | 11 | 62.3% | 0.539 |
| common_cohort_k3 | original_rf | 11 | 33.5% | 0.265 |
| common_cohort_k3 | random_forest | 11 | 61.0% | 0.531 |
| common_cohort_k4 | majority_stage | 11 | 35.4% | 0.104 |
| common_cohort_k4 | neural_network | 11 | 62.9% | 0.561 |
| common_cohort_k4 | original_rf | 11 | 37.1% | 0.309 |
| common_cohort_k4 | random_forest | 11 | 63.2% | 0.555 |
| common_cohort_k5 | majority_stage | 11 | 37.5% | 0.109 |
| common_cohort_k5 | neural_network | 11 | 59.1% | 0.521 |
| common_cohort_k5 | original_rf | 11 | 35.5% | 0.295 |
| common_cohort_k5 | random_forest | 11 | 60.8% | 0.517 |

The original RF baseline is retrained on exactly the same target training nights, using its original features/settings; it has no population training. Its contrast with the new models measures the entire pipeline change, not just the classifier. The two new models share all available training data.

The main cohort (all_subjects_personalized_k1_plus) averages every personalized history k>=1, giving each person equal weight: per-subject means first, then the subject mean. This cohort distinguishes anchored from walk_forward evaluation; the max-history cohort is identical in both because splits coincide at k=max. Only the common cohort supports a controlled 0-5-night curve. Participants, not epochs, are the statistical units. Test results must not be used to retune this experiment. No clinical reliability threshold or minimum-night claim is established.
