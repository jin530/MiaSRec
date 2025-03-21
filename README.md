# MiaSRec
This is the official code for SIGIR 2024 paper: 'Multi-intent-aware Session-based Recommendation'.

We implemented our model based on the recommendation framework library [RecBole v1.2.0](https://github.com/RUCAIBox/RecBole) and [CORE](https://github.com/RUCAIBox/CORE).

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
Please cite our paper:
```
@inproceedings{sigir/0001KCL24,
  author       = {Minjin Choi and
                  Hye{-}young Kim and
                  Hyunsouk Cho and
                  Jongwuk Lee},
  title        = {Multi-intent-aware Session-based Recommendation},
  booktitle    = {Proceedings of the 47th International {ACM} {SIGIR} Conference on
                  Research and Development in Information Retrieval, {SIGIR} 2024, Washington
                  DC, USA, July 14-18, 2024},
  pages        = {2532--2536},
  publisher    = {{ACM}},
  year         = {2024},
  doi          = {10.1145/3626772.3657928},
}
```
