# Texture InPainting by Gaussian model

We implemented the inpainting method for texture images introduced in

> *Microtexture inpainting through gaussian conditional simulation*, Bruno Galerne, Arthur Leclaire, Lionel Moisan

**Inpainting** is the task of **recovering** a missing region of an image. This region is called a **masked region**. The paper we followed propose an inpainting method in the case of texture images. In this framework texture images are supposed to be a **random gaussian field**. Authors used the work of Julesz, on the **translation invariance** of texture images statistics of order 1 and 2, to model and **estimate the variance** of this gaussian vector. Masked region is then recovered by **conditional simulation**.

## Short presentation of the method

<p align="left">
  <img src="img/mask.png" width="60%">
</p>

We model our pixels grid as a Markov random field. **Markov property** states that:

<p align="center">
  <img src="img/markov_hypo_eq.png" width="50%">
</p>

We will use the **Markov blanket** around the masked region in order to infer inpainted region (recovered region). Here is an example of a Markov blanket of one pixel in a Markov random field.

<p align="center">
  <img src="img/markov_blanket_draw.png" width="20%">
</p>

Following Julesz's theory it is assumed that statistics of order 1 and 2 are translation invariant in the case of texture images.

Here is an example where we compared statistics of order 1 and 2 for two images of pebbles. We can notice the similarity of pixel statistics for these two images.

<p align="left">
  <img src="img/stats_1.png" width="55%">
</p>

<p align="left">
  <img src="img/stats_2.png" width="55%">
</p>

Starting from a **Discret Spot Noise** (DSN) we can generate texture image by using the model:  

<p align="center">
  <img src="img/dsn_eq.png" width="40%">
</p>

We have the theorical guarantee from Central Limit Theorem that asymptotically it will follow a gaussian distribution. Hence we have the following result:

<p align="left">
  <img src="img/clt_eq.png" width="40%">
</p>
 
 Therefore we notice that we obtain a **stationary** gaussian process.  
 
 In the following we used this model to sample a texture image of paper by consecutive translation of spot noise (small patch from texture image).

<p align="center">
  <img src="img/paper_generated_1.png" width="32%">
  <img src="img/paper_generated_2.png" width="15%">
</p>

Finally we want to sample H = F* + G - G* where G ~ F and G independant of F. We will sample the **kriging estimator** F* and the **kriging residual or innovation component** G - G* using the **kriging coefficients**. In summary:

<p align="left">
  <img src="img/components_eq.png" width="60%">
</p>

And to compute the kriging coefficients we need to solve the **kriging system**:

<p align="left">
  <img src="img/kriging_eq_2.png" width="15%">
</p>
 
 where:
 
 <p align="left">
  <img src="img/kriging_eq_1.png" width="15%">
</p>

## Results

Without further ado here are our results:

### Wood

<p align="center">
  <img src="img/demo_wood_big_spot_1.png" width="39%">
  <img src="img/demo_wood_big_spot_2.png" width="18%">
</p>

### Leather

<p align="center">
  <img src="img/demo_leather.png" width="60%">
</p>

### Fur

<p align="center">
  <img src="img/demo_fur_masked.png" width="30%">
  <img src="img/demo_fur_result.png" width="30%">
</p>

### Brick

<p align="center">
  <img src="img/demo_brick_with_adsn.png" width="60%">
</p>

Here we noticed that it does not work since texture image contains **too much structure**. Hence it does match the invariance assumptions. Indeed we failed to sample a good DSN:

<p align="center">
  <img src="img/adsn_fail.png" width="30%">
</p>

Although the texture is so structured we could recover the missing region just by using the kriging estimator and we got a perfect result.

<p align="center">
  <img src="img/demo_brick_without_adsn.png" width="30%">
</p>

### Leaves

Obvious case of failure. Texture image contrains a highly complex structure. We are out of our framework. 

<p align="center">
  <img src="img/demo_leaves.png" width="60%">
</p>

### Pebbles

<p align="center">
  <img src="img/demo_pebble.png" width="65%">
</p>
