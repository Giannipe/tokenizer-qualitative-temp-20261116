# Appendix captions

## front_normalization_winners

Rows: RGB; argmax of competitive query responsibilities R; argmax of spatially normalized competitive pooling A; argmax of independent spatial softmax from the identical QK scores. Same early-BERT checkpoint and inputs. This isolates normalization, not retraining. Assignments use fixed opacity 0.55 in every row. Native 16×16 patch grid, display crop only.

## front_controlled_slot_maps

Rows: RGB; competitive maps for fixed queries [0, 4, 8, 12]; independent maps for the same numeric query indices. All use common log2 attention lift 1×–8× and opacity 0.65. Same QK scores; changing only normalization. The independent branch is a diagnostic counterfactual, not a separately trained policy.

## front_trained_models_slot_maps

Rows: RGB; competitive maps for fixed queries [0, 4, 8, 12]; independent maps for the same numeric query indices. All use common log2 attention lift 1×–8× and opacity 0.65. Real trained checkpoints: single-head early-BERT with competition versus 12-head VLI without competition, both final checkpoints. Architecture/interaction also differ, so this is not an isolated competitive-routing ablation. Maps are geometric pooling distributions; multi-head semantic pooling has additional head-specific weights. Query IDs are not aligned semantic identities across checkpoints.

## front_trained_models_winners

Rows: RGB; argmax of geometric pooling maps from the competitive single-head early-BERT model; argmax of geometric pooling maps from the noncompetitive 12-head VLI model. Both colored maps are post-hoc winners based on spatially normalized A, not a native R versus A comparison. Models differ in more than routing; query colors do not establish cross-model object correspondence.

## wrist_normalization_winners

Rows: RGB; argmax of competitive query responsibilities R; argmax of spatially normalized competitive pooling A; argmax of independent spatial softmax from the identical QK scores. Same early-BERT checkpoint and inputs. This isolates normalization, not retraining. Assignments use fixed opacity 0.55 in every row. Native 16×16 patch grid, display crop only.

## wrist_controlled_slot_maps

Rows: RGB; competitive maps for fixed queries [0, 4, 8, 12]; independent maps for the same numeric query indices. All use common log2 attention lift 1×–8× and opacity 0.65. Same QK scores; changing only normalization. The independent branch is a diagnostic counterfactual, not a separately trained policy.

## wrist_trained_models_slot_maps

Rows: RGB; competitive maps for fixed queries [0, 4, 8, 12]; independent maps for the same numeric query indices. All use common log2 attention lift 1×–8× and opacity 0.65. Real trained checkpoints: single-head early-BERT with competition versus 12-head VLI without competition, both final checkpoints. Architecture/interaction also differ, so this is not an isolated competitive-routing ablation. Maps are geometric pooling distributions; multi-head semantic pooling has additional head-specific weights. Query IDs are not aligned semantic identities across checkpoints.

## wrist_trained_models_winners

Rows: RGB; argmax of geometric pooling maps from the competitive single-head early-BERT model; argmax of geometric pooling maps from the noncompetitive 12-head VLI model. Both colored maps are post-hoc winners based on spatially normalized A, not a native R versus A comparison. Models differ in more than routing; query colors do not establish cross-model object correspondence.

Columns: task 1, ep 0, frame 105; task 1, ep 0, frame 223; task 2, ep 50, frame 116; task 2, ep 50, frame 247; task 3, ep 0, frame 89; task 3, ep 0, frame 245. All-slot atlases use IDs 0–15 in row-major order; no maps omitted. Query subset 0/4/8/12 chosen a priori, not ranked for favorable appearance.
