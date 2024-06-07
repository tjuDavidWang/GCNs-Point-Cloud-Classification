# 3D Point Cloud Classification with Multi-Layer GCN

Implementing 3D point cloud classification on [ModelNet40](https://modelnet.cs.princeton.edu/) using a multi-layer GCN network.

## Overview

<img src=".\imgs\framework.png" alt="framework" style="zoom:15%;" />

4 main modules：**Residual Block**, **EdgeConv Operation**, **Dynamic Graph Construction**,  and **Dilation Convolution**.

<img src=".\imgs\exp.png" alt="exp" style="zoom:5%;" />

## Dataset Preparation

Due to network connectivity issues, it is recommended to download the ModelNet40 dataset locally and then upload it to the server using `SCP` or similar tools.

Link: https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip

Extract it to the `./data` directory to create the `data/modelnet40_ply_hdf5_2048` folder.

## Environment Setup

My experimental environment includes:

- NVIDIA GeForce RTX 3090 GPUs
- CUDA Version: 12.1

You can set up the environment using the following commands:

```sh
conda create -n pointcloud python=3.9
conda activate pointcloud
pip install torch scikit-learn numpy h5py
# for DeepGCNs depend on your own version
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

For experiment visualization, I use wandb for real-time monitoring in `main.py`. Additionally, I use `tensorboard` for visualization in `main_tensorboard.py`.

```sh
pip install tensorboard  # TensorBoard
pip install wandb  # Weights and Biases
```

## Command Line Arguments

This section provides detailed explanations for the command line arguments used in the script. Each argument has a specific role and can be adjusted to control various aspects of the model training and evaluation process.

| Argument Name  | Type    | Description                                                  |
| -------------- | ------- | ------------------------------------------------------------ |
| `--exp_name`   | `str`   | Name of the experiment.                                      |
| `--model`      | `str`   | Model to use. Options: `pointnet`, `dgcnn`, `DeepGCN`, `GCN`, `EdgeGCN`. |
| `--dataset`    | `str`   | Dataset to use. Options: `modelnet40`.                       |
| `--batch_size` | `int`   | Size of the training batch.                                  |
| `--epochs`     | `int`   | Number of training epochs.                                   |
| `--lr`         | `float` | Learning rate for the optimizer.                             |
| `--eval`       | `bool`  | Evaluate the model if set to `True`.                         |
| `--model_path` | `str`   | Pretrained model path.                                       |
| `--k`          | `int`   | Number of nearest neighbors to use.                          |
| `--num_points` | `int`   | Number of points to use from each point cloud.               |
| `--n_blocks`   | `int`   | Number of basic blocks in the backbone.                      |
| `--n_filters`  | `int`   | Number of channels of deep features.                         |
| `--dynamic`    | `bool`  | Use dynamic adjacency matrix if set to `True`.               |
| `--dilated`    | `bool`  | Use dilated k-nearest neighbors if set to `True`.            |

The codes for models [PointNet](https://github.com/charlesq34/pointnet), [DGCNN](https://github.com/WangYueFt/dgcnn), and [DeepGCNs](https://github.com/lightaime/deep_gcns_torch) are from the respective repositories. The GCN model is based on the [GCN repository](https://github.com/tkipf/pygcn). EdgeGCN is my implementation.

### Example Usage

#### Training

```bash
python main.py
```

#### Testing

```bash
python main.py --eval True
```

## File Description

```python
EdgeGCN/
├── imgs/ 					# images shown on README
├── data/ 					# dataset
├── deep_gcns_torch/		# implement of DeepGCNs
├── pretrained/				# pretrained model
├── data.py					# dataloader
├── util.py					# IO toolkit
├── model.py				# model PointNet&DGCNN
├── DeepGCN.py				# model DeepGCN
├── GCN.py					# model GCN
├── EdgeGCN.py				# model EdgeGCN
├── main.py					# main with wandb
├── main_tensorboard.py		# main with tensorboard
└── README.md				# README
```

## License

This project is licensed under the MIT License.

## Acknowledgement

The structure of this codebase is borrowed from [DGCNN](https://github.com/WangYueFt/dgcnn).

