# Known Limitations

This note records the main limitations I need to acknowledge when writing up the project.

## Training and Evaluation

- V1-V3 reach 100% on the internal test split, which indicates that the curated dataset is much easier than the real-world problem.
- V4 improves generalisation, but its held-out external accuracy is 69.93% on 153 images rather than the higher 82% seen on the full external set.
- The full 762-image external result for V4 includes images used during fine-tuning, so it should not be presented as an independent final test result.

## Data

- The original non-psoriasis class is the main weakness in the project because it mostly contains clear healthy skin rather than realistic alternative lesions.
- The original dataset is curated rather than clinical, so strong internal performance does not translate directly to real-world use.
- External images come from mixed public sources and vary in quality, framing, lighting, and lesion type.
- Psoriasis subtype labels are too sparse for a reliable subtype classifier.

## Fairness

- Internal fairness analysis uses ITA as a proxy measure, not formal demographic metadata. ITA can be distorted by lesion colour, lighting, shadows, background pixels, and image compression. Internal group sizes are uneven, so those estimates should be treated as exploratory only.
- Fitzpatrick17k external fairness analysis uses real Fitzpatrick scale labels (Types I–VI) and is stronger evidence. V3 shows a sensitivity gap of 0.175 between the best (Type V, 0.9104) and worst (Type I, 0.7364) groups.
- The direction of the gap — higher sensitivity for darker skin types — is unexpected and not fully explained. It should be reported honestly rather than presented as evidence the model is fair for all groups.
- Fitzpatrick17k has only 24 Type VI (dark) psoriasis images, so the Type VI sensitivity estimate (0.875) is based on a small sample and has wide uncertainty.

## Deployment

- This is a research prototype and not a medical device.
- Grad-CAM highlights model attention, not medically validated diagnostic reasoning.
- No clinical validation, prospective testing, or specialist review has been carried out.

## Reproducibility

- Random seeds are fixed, but minor variation across devices and backends is still possible.
- The project is reproducible as a student research workflow, not as a locked clinical benchmark.
