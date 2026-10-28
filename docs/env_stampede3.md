# GrainGNN on TACC Stampede3 (H100)

**Prerequisite:** conda (Anaconda or Miniconda) installed under `$WORK`.

## Step 1. Create the environment

```sh
export PIP_CACHE_DIR=$SCRATCH/.cache/pip
conda create -n GNN --override-channels -c conda-forge python=3.10 pip
conda activate GNN
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
pip install torch-geometric==2.6.1
pip install -r requirements.txt
```

Do not conda-install anything else into `GNN`. `torch-sparse/scatter/cluster` are not needed.

## Step 2. Check

```sh
gzip -dk rawdat_PF/40_40/*.h5.gz
python3 graph_trajectory.py --mode=test --rawdat_dir=./rawdat_PF/40_40/ --seed=10020 --lxd=40 --save_dir=./graphs/40_40/
python3 test.py --truth_dir=./graphs/40_40/ --seed=10020
```

Expected: 72 / 75 grain elimination events, last-layer error 0.1128.

## Step 3. Submit a job

```sh
#!/bin/bash
#SBATCH -p h100
#SBATCH -N 1 -n 1 -t 01:00:00
#SBATCH -A ASC21034
source $WORK/anaconda3/bin/activate GNN
export PYTHONUNBUFFERED=1
cd $SCRATCH/graphGNN
python3 test.py --truth_dir=./graphs/40_40/ --seed=10020
```

## Step 4. Train

```sh
gzip -d dataset_train.pkl.gz        # keep on $SCRATCH (3.6 GB)
python3 train.py --model_type=regressor  --model_id=0 --device=cuda --model_dir=./model_demo/ > regressor0.log
python3 train.py --model_type=classifier --model_id=1 --device=cuda --model_dir=./model_demo/ > classifier1.log
python3 test.py --truth_dir=./graphs/40_40/ --seed=10020 --model_dir=./model_demo/
```

`--model_dir` other than `./model/` keeps the paper models intact.

## Versions

Same as Vista ([env_vista.md](env_vista.md)): python 3.10, torch 2.8.0+cu129, torch-geometric 2.6.1, numpy 1.26, scipy 1.15, matplotlib 3.10, h5py 3.16.
