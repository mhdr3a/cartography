# Dataset Cartography [Updated]

Code for the paper [Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics](https://aclanthology.org/2020.emnlp-main.746) at EMNLP 2020.

This repository contains implementation of data maps, as well as other data selection baselines, along with notebooks for data map visualizations.

If using, please cite:
```
@inproceedings{swayamdipta2020dataset,
    title={Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics},
    author={Swabha Swayamdipta and Roy Schwartz and Nicholas Lourie and Yizhong Wang and Hannaneh Hajishirzi and Noah A. Smith and Yejin Choi},
    booktitle={Proceedings of EMNLP},
    url={https://arxiv.org/abs/2009.10795},
    year={2020}
}
```
This repository can be used to build Data Maps, like [this one for SNLI using a RoBERTa-Large classifier](./sample/SNLI_RoBERTa.pdf).
![SNLI Data Map with RoBERTa-Large](./sample/SNLI_RoBERTa.png)

### Pre-requisites

This repository is based on the [HuggingFace Transformers](https://github.com/huggingface/transformers) library.
<!-- Hyperparameter tuning is based on [HFTune](https://github.com/allenai/hftune). -->

#### Initial Setup
```
!git clone https://github.com/mhdr3a/cartography
!mv /content/cartography /content/tmp
!mv /content/tmp/* /content/
!rm /content/tmp -r
!pip install -r /content/requirements.txt
```
#### Download and Prepare the MNLI Dataset
```
!wget https://dl.fbaipublicfiles.com/glue/data/MNLI.zip
!mkdir /content/data/ && mkdir /content/data/glue
!unzip /content/MNLI.zip -d /content/data/glue/
```

### Train GLUE-style model and compute training dynamics

To train a GLUE-style model using this repository:

```
python -m cartography.classification.run_glue \
    -c configs/$TASK.jsonnet \
    --do_train \
    --do_eval \
    -o $MODEL_OUTPUT_DIR
```
The best configurations for our experiments for each of the `$TASK`s (SNLI, MNLI, QNLI or WINOGRANDE) are provided under [configs](./configs).

This produces a training dynamics directory `$MODEL_OUTPUT_DIR/training_dynamics`, see a sample [here](./sample/training_dynamics/).

*Note:* you can use any other set up to train your model (independent of this repository) as long as you produce the `dynamics_epoch_$X.jsonl` for plotting data maps, and filtering different regions of the data.
The `.jsonl` file must contain the following fields for every training instance:
- `guid` : instance ID matching that in the original data file, for filtering,
- `logits_epoch_$X` : logits for the training instance under epoch `$X`,
- `gold` : index of the gold label, must match the logits array.

#### Train the RoBERTa-Base model with the MNLI training set for 6 epochs:
```
!python -m cartography.classification.run_glue \
    -c configs/mnli.jsonnet \
    --do_train \
    --do_eval \
    -o /content/results/
!zip -r results.zip /content/results/
!mkdir /content/drive/MyDrive/mnli-6
!mv /content/results.zip /content/drive/MyDrive/mnli-6/
```
#### Fine-tune the pretrained model, with the top 33% of the most ambiguous samples from MNLI training set for 3 epochs
```
!python -m cartography.classification.run_glue \
    -c configs/mnli.jsonnet \
    --do_train \
    --do_eval \
    -o /content/results/
!zip -r results.zip /content/results/
!mkdir /content/drive/MyDrive/mnli-6-var33-3
!mv /content/results.zip /content/drive/MyDrive/mnli-6-var33-3/
```
- Change the lines 10 and 16 of **mnli.jsonnet** to the following (respectively):
```
local DATA_DIR = "/content/pretrained/filtered/cartography_variability_0.33/" + TASK;
"model_name_or_path": "/content/results/",
```
- Do as the line 180 of **run_glue.py** suggests.
- To save the last checkpoint, instead of the one which results in the best dev performance, do as the lines 369-370 of **run_glue.py** suggest.


### Plot Data Maps

To plot data maps for a trained `$MODEL` (e.g. RoBERTa-Large) on a given `$TASK` (e.g. SNLI, MNLI, QNLI or WINOGRANDE):

```
python -m cartography.selection.train_dy_filtering \
    --plot \
    --task_name $TASK \
    --model_dir $PATH_TO_MODEL_OUTPUT_DIR_WITH_TRAINING_DYNAMICS \
    --model $MODEL_NAME
```
#### Plot the data map of the mnli-6 model
```
!python -m cartography.selection.train_dy_filtering \
    --plot \
    --task_name MNLI \
    --model_dir /content/pretrained/results \
    --model roberta-base
!mv /content/cartography/MNLI_roberta-base.pdf /content/drive/MyDrive/mnli-6
```

#### Data Map Coordinates

The coordinates for producing RoBERTa-Large data maps for SNLI, QNLI, MNLI and WINOGRANDE, as reported in the paper can be found under `data/data_map_coordinates/`. Each `.jsonl` file contains the following fields for each instance in the train set:
- `guid` : instance ID matching that in the original data file,
- `index`,
- `confidence`,
- `variability`,
- `correctness`.


### Data Selection

To select (different amounts of) data based on various metrics from training dynamics:

```
python -m cartography.selection.train_dy_filtering \
    --filter \
    --task_name $TASK \
    --model_dir $PATH_TO_MODEL_OUTPUT_DIR_WITH_TRAINING_DYNAMICS \
    --metric $METRIC \
    --data_dir $PATH_TO_GLUE_DIR_WITH_ORIGINAL_DATA_IN_TSV_FORMAT
```

Supported `$TASK`s include SNLI, QNLI, MNLI and WINOGRANDE, and `$METRIC`s include `confidence`, `variability`, `correctness`, `forgetfulness` and `threshold_closeness`; see [paper](https://aclanthology.org/2020.emnlp-main.746) for more details.


To select _hard-to-learn_ instances, set `$METRIC` as "confidence" and for _ambiguous_, set `$METRIC` as "variability". For _easy-to-learn_ instances: set `$METRIC` as "confidence" and use the flag `--worst`.

#### Filter out the most ambiguous samples from the MNLI training set
```
!python -m cartography.selection.train_dy_filtering \
    --filter \
    --task_name MNLI \
    --model_dir /content/pretrained/results \
    --metric variability \
    --data_dir /content/data/glue
!zip -r filtered-var.zip /content/filtered/
!mv /content/filtered-var.zip /content/drive/MyDrive/mnli-6/
```

### Contact and Reference

For questions and usage issues, please contact `swabhas@allenai.org`. If you use dataset cartography for research, please cite [our paper](https://aclanthology.org/2020.emnlp-main.746) as follows:

```
@inproceedings{swayamdipta-etal-2020-dataset,
    title = "Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics",
    author = "Swayamdipta, Swabha  and
      Schwartz, Roy  and
      Lourie, Nicholas  and
      Wang, Yizhong  and
      Hajishirzi, Hannaneh  and
      Smith, Noah A.  and
      Choi, Yejin",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.746",
    doi = "10.18653/v1/2020.emnlp-main.746",
    pages = "9275--9293",
}
```
Copyright [2020] [Swabha Swayamdipta]

