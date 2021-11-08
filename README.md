# CoNLL Noise Induction Experiment

## Aim
To investigate the effect of **annotator accuracy** on **model accuracy** on a Named Entity Recognition Task through artificial induction of noise in annotated labels.

More details can be found in the accompanying paper, [*pending anonymous review*].

## Instructions
**Note**: Some scripts in this repository require Python2. 

1. Install `requirements.txt`.
2. Download [glove.6B.300d.txt](https://nlp.stanford.edu/projects/glove/) and place into `data/glove.6B/`.
3. Download [CoNLL NER data](https://www.clips.uantwerpen.be/conll2003/ner/) and place into `data/ner-mturk/`.
4. Run `run_test.py` in Python2. 
5. Observe output data in `output/`.
6. Open `charts and plot.ipynb` and run cells to analyse and visualise output data.

## References
Data is from the [CoNLL-2003 shared task](https://www.clips.uantwerpen.be/conll2003/ner/). 

Model architecture adapted using [Rodrigues & Pereira's CrowdLayer model for NER](https://github.com/fmpr/CrowdLayer). More details can be found in the paper: 

> Rodrigues, F. and Pereira, F. Deep Learning from Crowds. In Proc. of the Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18).
