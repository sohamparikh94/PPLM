# CIS700

This repository is forked from the original PPLM repository (https://github.com/uber-research/PPLM). 

## Setup

```bash
pip install -r requirements.txt
```

## PPLM-BoW 

### Example command for bag-of-words control

```bash
python run_pplm.py -B military --cond_text "The potato" --length 50 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.03 --window_length 5 --kl_scale 0.01 --gm_scale 0.99 --colorama --sample --bow_type 1
```
If you want to run on multiple prompts, specify the ```filepath``` where the prompts are stored (one per line) and run the following command:

```bash
python run_pplm.py -B military --cond_text "The potato" --length 50 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.03 --window_length 5 --kl_scale 0.01 --gm_scale 0.99 --colorama --sample --bow_type 1 --multiple_prompts --prompts_file filepath
```

### Colab Notebook

The colab notebook for running our code is here: https://colab.research.google.com/drive/13dGRUyj3rxHP38yId4Azd5Hzmy33qHIG