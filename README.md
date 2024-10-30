# Simulation-Free Training of Neural ODEs on Paired Data
This repository contains the official implementation of the paper **"Simulation-Free Training of Neural ODEs on Paired Data (NeurIPS 2024)"** 

Semin Kim*, Jaehoon Yoo*, Jinwoo Kim, Yeonwoo Cha, Saehoon Kim, Seunghoon Hong

[Paper Link (TODO)](TODO)

## Setup
To set up the environment, start by installing dependencies listed in `requirements.txt`. You can also use Docker to streamline the setup process.

1. **Docker Setup:**
```
docker pull pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
docker run -it pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel bash
```

2. **Clone the Repository:**
```
git clone https://github.com/seminkim/simulation-free-node.git
```
3. **Install Requirements:**
```
pip install -r requirements.txt
```
## Datasets
Place all datasets in the `.data` directory. By default, this code automatically downloads the MNIST, CIFAR-10, and SVHN datasets into the `.data` directory.


The UCI dataset, composed of 10 tasks (`bostonHousing`, `concrete`, `energy`, `kin8nm`, `naval-propulsion-plant`, `power-plant`, `protein-tertiary-structure`, `wine-quality-red`, `yacht`, and `YearPredictionMSD`), can be manually downloaded from the **Usage** part of the following repository: [CARD](https://github.com/XzwHan/CARD). 

## Training
Scripts for training are available for both classification and regression tasks.

### Classification
To train a model for a classification task, run:
```
python main.py fit --config configs/{dataset_name}.yaml --name {exp_name}
```

### Regression
For regression tasks (only supported with UCI datasets), use the following command:

```
python main.py fit --config configs/uci.yaml --name {exp_name} --data.task {task_name} --data.split_num {split_num}
```
In this command, specify the UCI task name and the data split number accordingly.

## Inference
Use the following commands for model evaluation.
### Classification
```
python main.py validate --config configs/{dataset_name}.yaml --name {exp_name} --ckpt_path {ckpt_path}
```

### Regression
For UCI regression tasks:

```
python main.py validate --config configs/uci.yaml --name {exp_name} --data.task {task_name} --data.split_num {split_num} --ckpt_path {ckpt_path}
```

### Checkpoints
Trained checkpoints can be found at release tab of this repository.

|Dataset    |Dopri Acc. |Link       |
|:---:      |:---:      |:---:      |
|MNIST      |99.30%     |[TODO]()|
|SVHN       |96.12%     |[TODO]()|
|CIFAR10    |88.89%     |[TODO]()|


## Additional Notes
### Logging
We use wandb to monitor training progress and inference results.
The wandb run name will match the argument provided for `--name`.
You can also change the project name by modifying `trainer.logger.init_args.project` in the configuration file (default value is `SFNO_exp`).

### Running Your Own Experiment
Our code is implented with `LightningCLI`, so you can simply overwrite the config via command-line arguments to experiment with various settings.

Examples:
```
# Run MNIST experiment with batch size 128
python main.py fit --config configs/mnist.yaml --name mnist_b128 --data.batch_size 128

# Run SVHN experiment with explicit sampling of $t=0$ with probability 0.01
python main.py fit --config configs/svhn.yaml --name svhn_zero_001 --model.init_args.force_zero_prob 0.01

# Run CIFAR10 experiment with 'concave' dynamics 
python main.py fit --config configs/cifar10.yaml --name cifar10_concave --model.init_args.dynamics concave
```
Refer to [Lightning Trainer documentation](https://lightning.ai/docs/pytorch/stable/common/trainer.html) for controlling trainer-related configurations (e.g., training steps or logging frequency).

## Acknowledgements
This implementation of this code was based on the following repositories: [NeuralODE](https://github.com/rtqichen/torchdiffeq), [ANODE](https://github.com/EmilienDupont/augmented-neural-odes), and [CARD](https://github.com/XzwHan/CARD).



## Citation
```
(TODO)
```