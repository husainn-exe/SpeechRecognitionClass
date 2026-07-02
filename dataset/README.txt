Put your audio files here:

1. dataset/raw/wake_word_1hour.wav
   - 1 hour audio containing at least 100 occurrences of "Hey, Jarvis".

2. dataset/raw/non_wake_word_1hour.wav
   - 1 hour audio without the wake word.

3. dataset/annotations.csv
   - Already included in this package from your uploaded annotation file.
   - Required columns:
     filename,start_time,end_time,label,speaker,environment

The preprocessing script will create:
- dataset/processed/wake_word/*.wav
- dataset/processed/non_wake_word/*.wav
- dataset/processed/metadata_segments.csv
