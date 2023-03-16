This branch contains the source code for the MT course project "Role of relative positional encoding in mitigating the length-based overfitting in Transformers".

This project was aimed at further investigating the [length-based overfitting in Transformer models](https://aclanthology.org/2021.emnlp-main.650). The hypothesis was that relative positional encoding might reduce the overfitting. I tested two approaches to relative positional encoding:
* [Self-Attention with Relative Position Representations](https://aclanthology.org/N18-2074/)
* [RNN as the position embedder](https://aclanthology.org/K19-1031/)

None of these positional encoding approaches helped to alleviate this issue. Future work can include testing other methods.

Reproducing the results of the NMT experiments:
1. Download CzEng 2.0
2. Download and preprocess the dataset using `custom_examples/translation/prepare-wmt20*` scripts.
3. Preprocess the datasets via Fairseq using `wrappers/preprocess_length_domain_translation.sh`
4. Run experiments using `run_rel_pos_enc_experiments.sh`
