# Noisy Label Detection

## Install Environment

### Virtual Environment

```bash
mkdir venv
python3 -m venv venv
source ./venv/bin/activate
```

### Get Project

```bash
git clone https://github.com/springfieldsr/capstone
```

### Install  dependencies package

To install PyTorch in different environments [this](https://pytorch.org/get-started/locally/)

```bash
pip install -r requirements.txt
```

## Execution

```bash
usage: image_cls.py [-h] [--datasets {CIFAR10,CIFAR100, MNIST}] [--models {resnet18, resnet101}]\
                              [--bs {1,4,16,32,64,256,1024}] [--epochs EPOCHS] [--lr LR]\
                              [--sp shuffle_percentage] [--es early_stop] [--k noise rate]
usage: text_cls.py [-h] [--datasets {bbc_text, 20news}]\
                              [--bs {1,4,16,32,64,256,1024}] [--epochs EPOCHS] [--lr LR]\
                              [--sp shuffle_percentage] [--es early_stop] [--k noise rate]
.

optional arguments:
  -h, --help            show this help message and exit
  --dataset {CIFAR10,CIFAR100}
  --model {resnet18,}  datasets
  --bs {1,4,16,64,256,1024}
                        batch_size
  --epochs EPOCHS       epochs of training
  --lr LR               learning rate
  --k                   track top k percentage of samples with highest loss
  --rp                  float between [0, 1] to specify beginning of loss recording at epoch: rp * optimum_epochs
  --ls                  wheter to shuffle labels of k percent of training samples

```