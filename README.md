# MiaSRec
This is the official code for SIGIR 2024 paper: 'Multi-intent-aware Session-based Recommendation'.

We implemented our model based on the recommedndation framework library [RecBole v1.2.0)](https://github.com/RUCAIBox/RecBole) and [CORE](https://github.com/RUCAIBox/CORE).

## Requirements

you can use the following command to install the environment
```bash
conda create -n miasrec python=3.8
conda activate miasrec
pip install -r requirements.txt
```

## Datasets
make `dataset` folder and unzip $DATASET$.zip to `dataset` folder
$DATASET$: (`diginetica`, `retailrocket`, `yoochoose`, `dressipi`, `tmall`, `lastfm`)
```bash
for DATASET in diginetica retailrocket yoochoose dressipi tmall lastfm
do
unzip $DATASET.zip -d dataset/$DATASET
done
```

## Reproduction

```bash
python main.py --model miasrec --dataset diginetica --beta_logit 0.9
python main.py --model miasrec --dataset retailrocket --beta_logit 0.8
python main.py --model miasrec --dataset yoochoose --beta_logit 0.7
python main.py --model miasrec --dataset tmall --beta_logit 0.9
python main.py --model miasrec --dataset dressipi --beta_logit 0.9
python main.py --model miasrec --dataset lastfm --beta_logit 0.9
```

## Citation
Please cite our papaer:
```
@article{choi2024multi,
  title={Multi-intent-aware Session-based Recommendation},
  author={Choi, Minjin and Kim, Hye-young and Cho, Hyunsouk and Lee, Jongwuk},
  journal={arXiv preprint arXiv:2405.00986},
  year={2024}
}
```