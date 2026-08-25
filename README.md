# StateFlowDiff

Clean reproducible implementation of the StateFlowDiff traffic-flow forecasting model.

## Layout

- `StateFlowDiff/`: model, training, data loading, diffusion, SFCN, LSTDE, HTRC, and DCA code.
- `configs/fujian30/`: StateFlowDiff-only YAML configurations and ablation configurations.
- `datasets/fujian-30/`: Fujian-30 data.
- `datasets/original_pems/`: the three original PeMS datasets.
- `datasets/pems_r30/`: the fifteen final R30 dataset splits.
- `plots/`: final paper PDFs and one final plotting script per figure.

Dataset-construction/splitting utilities and baseline implementations are excluded.

## Run

From the repository root:

```bash
python -m StateFlowDiff.train --config configs/fujian30/stateflowdiff_h12.yaml
```

The model alias preserves the original parameter/module graph so seeded training remains reproducible.
