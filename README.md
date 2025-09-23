# Long-Context Audio-Sheet Music Retrieval
This repository contains the code for the paper [Passage Summarization with recurrent models for Audio–Sheet Music Retrieval](https://arxiv.org/abs/2309.12111) presented at [ISMIR 2023](https://ismir2023.ismir.net/).

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/luisfvc/lcasr.git
   cd lcasr
   ```

2. Install dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate lcasr
   ```
## Citation
If you use this code in your research, please cite our paper:
```bibtex
@inproceedings{CarvalhoW23_A2S_Recurrent_ISMIR,
  title        = {Passage Summarization with Recurrent Models for Audio-Sheet Music Retrieval},
  author       = {Lu{\'i}s Carvalho and Gerhard Widmer},
  year         = 2023,
  booktitle    = {Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)},
  address      = {Milan, Italy}
}
```
## Related Work

This work builds upon our previous research in audio-sheet music retrieval:
- [Learning audio–sheet music correspondences (TISMIR 2018)](https://github.com/CPJKU/audio_sheet_retrieval)
- [Attention models for tempo-invariant retrieval (ISMIR 2019)](https://github.com/CPJKU/audio_sheet_retrieval/tree/ismir-2019)
- [Exploiting temporal dependencies (EUSIPCO 2021)](https://github.com/CPJKU/audio_sheet_retrieval/tree/eusipco-2021)

## Acknowledgements

This work is supported by the European Research Council (ERC) under the EU’s Horizon 2020 research and innovation programme,
grant agreement No. 101019375 (“Whither Music?”), and the Federal State of Upper Austria (LIT AI Lab).

## Contact
For questions or issues, you can contact me here or via [email](mailto:luisfeliperj90@gmail.com).
