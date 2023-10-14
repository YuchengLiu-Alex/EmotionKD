# EmotionKD
# EmotionKD

EmotionKD: A Cross-Modal Knowledge Distillation Framework for Emotion Recognition Based on Physiological Signals


![model_architecture](figures/pipeline.png)


<!-- # References
EmotionKD: A Cross-Modal Knowledge Distillation Framework for Emotion Recognition Based on Physiological Signals. (ACM MM 2023)

```latex
@inproceedings{ijcai2020-184,
  title     = {GraphSleepNet: Adaptive Spatial-Temporal Graph Convolutional Networks for Sleep Stage Classification},
  author    = {Jia, Ziyu and Lin, Youfang and Wang, Jing and Zhou, Ronghao and Ning, Xiaojun and He, Yuanlai and Zhao, Yaoshuai},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  pages     = {1324--1330},
  year      = {2020},
  month     = {7},
  doi       = {10.24963/ijcai.2020/184},
  url       = {https://doi.org/10.24963/ijcai.2020/184},
}
``` -->

# Datasets

 We evaluate our model on the DEAP and HCI-Tagging dataset. The DEAP and HCI-Tagging are open-access and collaborative database of laboratory-based polysomnography (PSG) recordings. Information on how to obtain it can be found in [DEAP](https://deap.readthedocs.io/en/master/) and [HCI-Tagging](https://mahnob-db.eu/hci-tagging/) specifically.

# Requirements

- Python 3.8
- Tensorflow 2.5.0
- numpy 1.19.5
- scipy 1.9.3
- scikit-learn 1.1.2

# Usage

We provide the distillation file `Buildmodel.py`. You need to extract the intermediate features as the inputs of the SleepKD distillation layer.

