# Encoding short music videos with MERT and [CLMR](https://github.com/Spijkervet/CLMR)

## Getting data
1. A large dataset of similar tracks was generated through Spotify user's playlists, Last.fm's get_similar API, and tracks of the same album `data/get_dataset.py`

## Evaluation
2. Base MERT and CLMR models was evaluated with purity scores (`train_test/eval.py`) and an interactive tSNE visualization of encoded tracks `train_test/generate_tsne_report.py`, allowing the user to hover over clusters and hear samples of nearby songs

## Running
