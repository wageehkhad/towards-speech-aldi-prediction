# Assets

This directory holds manifests and metadata only. Audio and model checkpoints
are not redistributed here. Obtain the corpora from their original sources and
lay them out as below; the scripts expect these paths by default.

```text
assets/
  models/
    whisper_aldi_sada_full_medium/    direct Whisper-medium ALDi
    mms1b_aldi_sada/                  direct MMS-1B ALDi
    whisper_sada_asr_full/            SADA-finetuned Whisper ASR (cascaded baseline)
  sada/
    manifest_train_aldi.csv.gz        shipped, gunzip before training
    manifest_val_aldi.csv             shipped
    manifest_test_aldi.csv            shipped
    wavs/                             SADA audio
  casablanca/
    casablanca_aldi_scored.csv        shipped
    casablanca_audio_aldi/            HF dataset saved with save_to_disk()
  mediaspeech/
    manifest_mediaspeech_ar_aldi.csv  shipped
  genre_based_aldi/
    sources.csv                       shipped
    raw_audio/                        trimmed 16 kHz mono WAVs, one per source
    clips/                            15-second segments
```

Manifests ending in `_aldi` carry the silver-standard target scores, produced by
running `score_sada_aldi.py` over the manual transcripts.

## Sources

- SADA: https://www.kaggle.com/datasets/sdaiancai/sada2022
- Casablanca: https://huggingface.co/datasets/UBC-NLP/Casablanca
- MediaSpeech: https://www.openslr.org/108/
