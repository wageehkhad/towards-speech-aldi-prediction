# Error Analysis Summary (Whisper-medium ALDi)

- Checkpoint used: `assets/models/whisper_aldi_sada_full_medium/checkpoint_epoch5.pt`

## Cross-dataset summary

| dataset | n | mae | mse | pearson | spearman | mean_prediction | mean_ground_truth | prediction_std | error_mean | error_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| casablanca | 6819 | 0.1429 | 0.0395 | 0.6796 | 0.6216 | 0.7186 | 0.7560 | 0.2247 | -0.0374 | 0.1951 |
| mediaspeech | 2505 | 0.0766 | 0.0131 | 0.8628 | 0.7044 | 0.1378 | 0.1548 | 0.1927 | -0.0169 | 0.1130 |
| sada_test | 6193 | 0.1295 | 0.0364 | 0.8254 | 0.8176 | 0.6281 | 0.6250 | 0.3011 | 0.0032 | 0.1908 |

- Best correlation dataset: `mediaspeech` (Pearson=0.8628, Spearman=0.7044)
- Highest-error dataset: `casablanca` (MAE=0.1429)

## Hardest dialects (by MAE)

| dataset | dialect | n | mae | pearson | spearman |
| --- | --- | --- | --- | --- | --- |
| casablanca | Mauritania | 953 | 0.1818 | 0.5527 | 0.4851 |
| casablanca | Algeria | 844 | 0.1618 | 0.5833 | 0.5126 |
| casablanca | Egypt | 846 | 0.1502 | 0.6535 | 0.6271 |
| casablanca | Yemen | 803 | 0.1488 | 0.7207 | 0.6551 |
| casablanca | Morocco | 1045 | 0.1385 | 0.5932 | 0.5202 |

## Duration effects

- `casablanca`: MAE 0-5s=0.1554, 20-30s=0.0834
- `sada_test`: MAE 0-5s=0.1538, 20-30s=0.0644

## ALDi-range effects

- `casablanca`: MAE MSA-ish(0-0.2)=0.2382, heavy dialect(0.8-1.0)=0.1227
- `mediaspeech`: MAE MSA-ish(0-0.2)=0.0585, heavy dialect(0.8-1.0)=0.1563
- `sada_test`: MAE MSA-ish(0-0.2)=0.1829, heavy dialect(0.8-1.0)=0.0972

## Worst-case examples

- Top worst rows saved to `worst_predictions.csv` (50 rows).

## Notes

- SADA manifest does not include explicit dialect labels; dialect is set to `SADA`.
- MediaSpeech is treated as `MSA` dialect in by-dialect grouping.
