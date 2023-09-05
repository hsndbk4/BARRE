# On the Robustness of Randomized Ensembles to Adversarial Perturbations

This repository contains the code for our paper [On the Robustness of Randomized Ensembles to Adversarial Perturbations](https://arxiv.org/abs/2302.01375) by Hassan Dbouk and [Naresh R. Shanbhag](http://shanbhag.ece.illinois.edu/) (ICML 2023).

# Running This Repo
This code was run with the following dependencies, make sure you have the appropriate versions downloaded and installed properly.
 ```
python 3.6.9
PyTorch 1.7.0
numpy 1.19.2
torchvision 0.8.0
```

1.  clone the repo: `git clone https://github.com/hsndbk4/BARRE.git`
2.  make sure the appropriate dataset folders are setup properly (check `get_dataloaders` in `datasets.py`)
3.  download a BARRE-trained REC of ResNet-20s on CIFAR-10 from [here](https://uofi.box.com/s/yrj4yw9woqznvnec858pvcreojld32bf)
4.  place the models in an appropriate folder in the root directory, e.g. `res20_cifar10_M5`

We are now set to run some scripts. To re-produce the ResNet-20 $\ell_\infty$ numbers in Table 1, you can run the following commands:

In order to evaluate the robustness of the trained models, please run:
```
python eval_robustness.py --M 5 --model res20 --batch_size 512 --sourcedir "res20_cifar10_M5" --outdir "res20_cifar10_M5" --normalize --use_osp
```
In order to re-produce the training outcome, please run:
```
python train_barre.py --M 5 --other_weight 1 --model res20 --batch_size 256 --outdir "res20_cifar10_M5" --normalize --osp_data_len 4096 --osp_batch_size 1024
```

## Citation

If you find our work helpful, please consider citing it.
```
@inproceedings{dbouk2023robustness,
  title={On the Robustness of Randomized Ensembles to Adversarial Perturbations},
  author={Dbouk, Hassan and Shanbhag, Naresh},
  booktitle={International Conference on Machine Learning},
  pages={7303--7328},
  year={2023},
  organization={PMLR}
}
```

## Acknowledgements

This work was supported by the Center for the Co-Design of Cognitive Systems (CoCoSys) funded by the Semiconductor Research Corporation (SRC) and the Defense Advanced Research Projects Agency (DARPA), and SRC’s Artificial Intelligence Hardware (AIHW) program.

Parts of the code in this repository are based on following public repositories:

* [https://github.com/zdhNarsil/margin-boosting](https://github.com/zdhNarsil/margin-boosting)
