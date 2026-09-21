# MangoLeafBD Dataset Augmentation Strategy

This project applies augmentation with a split-aware workflow to avoid train/validation/test leakage.

## Augmentation Types and Parameters

The following augmentations are defined and applied with conservative ranges to reduce artifact risk:

1. Rotation

- Random angle: `-30` to `+30` degrees
- Interpolation: bicubic
- Output size: unchanged (`expand=False`)

2. Horizontal Flip

- Probability: `1.0` for the generated horizontal-flip variant

3. Scaling

- Random scale factor: `0.9` to `1.1`
- Geometry handling: center crop or pad back to original size

4. Random Crop

- Random crop scale: `0.8` to `1.0` of original size
- Resized back to original dimensions

5. Color Adjustment

- Brightness factor: `0.8` to `1.2`
- Contrast factor: `0.8` to `1.2`
- Saturation factor: `0.8` to `1.2`

## Uniformity and Class Balance

- The exact same augmentation set is used for every class.
- Each image receives the same number of augmentation variants per type.
- This keeps augmentation balanced across categories and avoids over-augmenting selected classes.

## Split-Aware Processing

- The script first creates class-wise train/val/test splits.
- Originals are copied by split.
- Augmentation is then performed independently per split.
- Default behavior augments only train split (`--augment-splits train`) to preserve fair evaluation.
- If needed, val/test can be augmented independently (for ablation studies), e.g. `--augment-splits train val test`.

## Output Folder Design

Under the chosen output directory:

- `train/original/<class_name>/...`
- `train/rotation/<class_name>/...`
- `train/horizontal_flip/<class_name>/...`
- `train/scaling/<class_name>/...`
- `train/random_crop/<class_name>/...`
- `train/color_adjustment/<class_name>/...`
- Similar paths for `val` and `test`

A summary file is generated at:

- `augmentation_summary.json`

This includes split counts, augmentation counts, selected parameters, and random seed.
