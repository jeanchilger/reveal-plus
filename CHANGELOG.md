# General Notes
- `PURPOSE` variable removed;
- The `baseline` and `debug` sub directories were removed;
- `CORPLIST` was removed;

# Variables
- `NODCS` → `TOTAL_DOCUMENTS`;
- `NDUN` → `LABELED_DOCUMENTS`;
- `L` → `SAMPLE_SIZE`;
- `b` → `sub_sample`;
- `flag_discretize` → `discretize`;
- `NotRel` → `not_relevants`;
- `Rel` → `relevants`;
- `N` → `round`;
- `CORP` → `JUDGECLASS`;
- `RELTRAINDOC` → `REL_ON_TRAINSET`;
- `NOTRELTRAINDOC` → `NOT_REL_ON_TRAINSET`;
- `PREVALENCERATE` → `PREVALENCE_RATE`;
- `RELFINDDOC` → `rel_found_sample`;
- `RELRATE` → `rel_rate`;
- `CURRENTREL` → `current_rel`;
- `alreadyLabeledDocs` → `already_labeled_docs`;
- `allDocs` → `all_docs`;
- `Estimate` → `estimate` (only in `training` file for now);

# Files
- `x` → `intermediate_seed`;
- `saida_classificador` → `classifier_output`;
- `sub_new_positivo[0-9][0-9]` → `sub_new_positives[0-9][0-9]`;
- `sub_new_negativo[0-9][0-9]` → `sub_new_negatives[0-9][0-9]`;
- `trainsetB.[0-9][0-9].TOPIC` → `trainset_labels.[0-9][0-9].TOPIC`;
- `seed_ssarpB.TOPIC` → `seed_ssarp_labels.TOPIC`;
