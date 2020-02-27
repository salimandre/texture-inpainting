# Texture InPainting by Gaussian model

We implemented the inpainting method for texture images introduced in

> *Microtexture inpainting through gaussian conditional simulation*, Bruno Galerne, Arthur Leclaire, Lionel Moisan

**Inpainting** is the task of **recovering** a missing region of an image. This region is called a **masked region**. The paper we followed propose an inpainting method in the case of texture images. In this framework texture images are supposed to be a **random gaussian field**. Authors used the work of Julesz, on the **translation invariance** of texture images statistics of order 1 and 2, to model and **estimate the variance** of this gaussian vector. Masked region is then recovered by **conditional simulation**.

## Short presentation of the method

<p align="left">
  <img src="img/mask.png" width="60%">
</p>

<p align="left">
  <img src="img/components_eq.png" width="60%">
</p>

<p align="left">
  <img src="img/clt_eq.png" width="60%">
</p>

<p align="left">
  <img src="img/dsn_eq.png" width="60%">
</p>

<p align="left">
  <img src="img/markov_hypo_eq.png" width="60%">
</p>

<p align="left">
  <img src="img/markov_blanket_draw.png" width="30%">
</p>

<p align="left">
  <img src="img/stats_1.png" width="30%">
</p>

<p align="left">
  <img src="img/stats_2.png" width="30%">
</p>

krigging solution of ordinary kriging system 
 
## Results
