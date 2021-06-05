[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
<!-- [![image title][views-url] -->

# AnimeGAN - Deep Convolutional Generative Adverserial Network

PyTorch implementation of DCGAN introduced in the paper: [Unsupervised Representation Learning with Deep Convolutional 
Generative Adversarial Networks](https://arxiv.org/abs/1511.06434), Alec Radford, Luke Metz, Soumith Chintala.

<p align="center">
<img src="assets/outgif.gif" title="Generated Data Animation" alt="Generated Data Animation">
</p>

## Abstract

In recent years, supervised learning with convolutional networks (CNNs) has
seen huge adoption in computer vision applications. Comparatively, unsupervised
learning with CNNs has received less attention. In this work we hope to help
bridge the gap between the success of CNNs for supervised learning and unsupervised learning. We introduce a class of CNNs called deep convolutional generative
adversarial networks (DCGANs), that have certain architectural constraints, and
demonstrate that they are a strong candidate for unsupervised learning. Training
on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to
scenes in both the generator and discriminator. Additionally, we use the learned
features for novel tasks - demonstrating their applicability as general image representations..

**Generator architecture of DCGAN**

<p align="center">
<img src="assets/DCGAN.png" title="DCGAN Generator" alt="DCGAN Generator">
</p>

## Directory Structre

```
.
├── assets
├── data
├── docs
├── logs
├── pipelines
├── research
├── src
│   ├── Data.py
│   └── model.py
├── tests
├── weights
├── LICENSE
├── README.md
├── requirements.txt
└── train.py

```

## Run Training

```sh
python train.py \
    --wandbkey={{WANDB KEY}} \
    --projectname=AnimeGAN \
    --wandbentity={{WANDB USERNAME}} \
    --tensorboard=True \
    --dataset=anime \
    --kaggle_user={{KAGGLE USERNAME}} \
    --kaggle_key={{KAGGLE API KEY}} \
    --batch_size=32 \
    --epoch=5 \
    --load_checkpoints=True \
```


## References
1. **Alec Radford, Luke Metz, Soumith Chintala.** *Unsupervised representation learning with deep convolutional 
generative adversarial networks.*[[arxiv](https://arxiv.org/abs/1511.06434)]
2. **Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, 
Sherjil Ozair, Aaron Courville, Yoshua Bengio.** *Generative adversarial nets.* NIPS 2014 [[arxiv](https://arxiv.org/abs/1406.2661)]
3. **Ian Goodfellow.** *Tutorial: Generative Adversarial Networks.* NIPS 2016 [[arxiv](https://arxiv.org/abs/1701.00160)]
4. DCGAN Tutorial. [https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html]
5. PyTorch Docs. [https://pytorch.org/docs/stable/index.html]



<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->




<!-- MARKDOWN LINKS & IMAGES --> 
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/rohitkuk/AnimeGAN.svg?style=for-the-badge
[contributors-url]: https://github.com/rohitkuk/AnimeGAN/graphs/contributors
[views-url]: https://github.com/rohitkuk/AnimeGAN/graphs/traffic
[forks-shield]: https://img.shields.io/github/forks/rohitkuk/AnimeGAN.svg?style=for-the-badge
[forks-url]: https://github.com/rohitkuk/AnimeGAN/network/members
[stars-shield]: https://img.shields.io/github/stars/rohitkuk/AnimeGAN.svg?style=for-the-badge
[stars-url]: https://github.com/rohitkuk/AnimeGAN/stargazers
[issues-shield]: https://img.shields.io/github/issues/rohitkuk/AnimeGAN.svg?style=for-the-badge
[issues-url]: https://github.com/rohitkuk/AnimeGAN/issues
[license-shield]: https://img.shields.io/github/license/rohitkuk/AnimeGAN.svg?style=for-the-badge
[license-url]: https://github.com/rohitkuk/AnimeGAN/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/rohitkuk
[product-screenshot]: images/screenshot.png