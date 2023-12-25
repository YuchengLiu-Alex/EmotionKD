# EmotionKD

EmotionKD: A Cross-Modal Knowledge Distillation Framework for Emotion Recognition Based on Physiological Signals


![model_architecture](figures/pipeline.png)


# Datasets

 We evaluate our model on the DEAP and HCI-Tagging dataset. The DEAP and HCI-Tagging are open-access database of laboratory-based physiological recordings. Information on how to obtain it can be found in [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) and [HCI-Tagging](https://mahnob-db.eu/hci-tagging/) specifically.

# Requirements

- Python 3.8
- Tensorflow 2.5.0
- numpy 1.19.5
- scipy 1.9.3
- scikit-learn 1.1.2

# Webpage

The webpage is shown in [Here](https://yuchengliu-alex.github.io/EmotionKD/).

# References
EmotionKD: A Cross-Modal Knowledge Distillation Framework for Emotion Recognition Based on Physiological Signals. (ACM MM 2023)

```latex
            @inproceedings{10.1145/3581783.3612277,
                author = {Liu, Yucheng and Jia, Ziyu and Wang, Haichao},
                title = {EmotionKD: A Cross-Modal Knowledge Distillation Framework for Emotion Recognition Based on Physiological Signals},
                year = {2023},
                isbn = {9798400701085},
                publisher = {Association for Computing Machinery},
                address = {New York, NY, USA},
                url = {https://doi.org/10.1145/3581783.3612277},
                doi = {10.1145/3581783.3612277},
                abstract = {Emotion recognition using multi-modal physiological signals is an emerging field in affective computing that significantly improves performance compared to unimodal approaches. The combination of Electroencephalogram(EEG) and Galvanic Skin Response(GSR) signals are particularly effective for objective and complementary emotion recognition. However, the high cost and inconvenience of EEG signal acquisition severely hinder the popularity of multi-modal emotion recognition in real-world scenarios, while GSR signals are easier to obtain. To address this challenge, we propose EmotionKD, a framework for cross-modal knowledge distillation that simultaneously models the heterogeneity and interactivity of GSR and EEG signals under a unified framework. By using knowledge distillation, fully fused multi-modal features can be transferred to an unimodal GSR model to improve performance. Additionally, an adaptive feedback mechanism is proposed to enable the multi-modal model to dynamically adjust according to the performance of the unimodal model during knowledge distillation, which guides the unimodal model to enhance its performance in emotion recognition. Our experiment results demonstrate that the proposed model achieves state-of-the-art performance on two public datasets. Furthermore, our approach has the potential to reduce reliance on multi-modal data with lower sacrificed performance, making emotion recognition more applicable and feasible. The source code is available at https://github.com/YuchengLiu-Alex/EmotionKD},
                booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
                pages = {6122–6131},
                numpages = {10},
                keywords = {knowledge distillation, emotion recognition, galvanic skin response, affective computing, electroencephalogram},
                location = {Ottawa ON, Canada},
                series = {MM '23}
                }
```