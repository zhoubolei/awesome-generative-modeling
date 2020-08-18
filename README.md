# Papers on generative modeling 

## Well, check out our GenForce work first!

[GenForce](https://genforce.github.io): may generative force be with you. Refer to this page for our latest work.

CVPR2020: **Interpreting the Latent Space of GANs for Semantic Face Editing**. [paper](https://genforce.github.io/interfacegan) <br>
*comment*: use facial classifiers to discover the interpretable dimensions emerged in the GANs trained to synthesize faces.

CVPR2020: **Image Processing Using Multi-Code GAN Prior**. [paper](https://genforce.github.io/mganprior/) <br>
*comment*: use the pretrained GAN model as a representation prior to facilitate a series of image processing tasks, such as colorization, super-resolution, denoising.

ECCV2020: **In-Domain GAN Inversion for Real Image Editing**. [paper](https://genforce.github.io/idinvert/) <br>
*comment*: Inverted code from Image2StyleGAN does not have enough manipulatability. This work proposes to use an encoder to regularize the optimization to preserve the manipulatability. A novel semantic difusion application is also proposed. 

arXiv: **Closed-Form Factorization of Latent Semantics in GANs**. [paper](https://genforce.github.io/sefa/)<br>
*comment*: Unsupervised discovery of the interpretable dimensions in learned GAN models. The algorithm works blazingly fast with only 1 second!

arXiv: **Generative Hierarchical Features from Synthesizing Images**. [paper](https://genforce.github.io/ghfeat/) <br>
*comment*: It considers the pretrained StyleGAN model as a learned loss (similar to perceptual loss using learned VGG) to train a hierarchical encoder. This work calls to explore various applications of the encoder for both for discriminative tasks and generative tasks. It also echoes my opinion that ImageNet classification is NOT the only way to evaluate the merits of learned features from self-supervised learning. There are so many visual tasks out there, why just stick to dog
classification on ImageNet?? (fun fact: there are about 150 dog classes out of the 1000 ImageNet classes).

## Latent code to Image (StyleGAN and BigGAN types)

ECCV2020: **StyleGAN2 Distillation for Feed-forward Image Manipulation**. [paper](https://arxiv.org/pdf/2003.03581.pdf), [code?](https://github.com/EvgenyKashin/stylegan2-distillation)<br>
*comment*: use InterfaceGAN pipeline to generate paired data from pretrained styleGAN, then use the paired data to train a pix2pix network. 

ECCV2020: **Rewriting a Deep Generative Model**. [paper](https://rewriting.csail.mit.edu/)<br>
*comment*: interactively replace the unit semantic patterns. David is the guru of interface, always. 

ECCV2020: **Exploiting Deep Generative Prior for Versatile Image Restoration and Manipulation**. [paper](https://xingangpan.github.io/projects/DGP.html) <br>
*comment*: similar to the following SIGGRAPH'19 work, it develops an inversion method for finetuing the pretrained GAN weight to invert an image, then applies to a series of image processing tasks (similar to [mGAN prior](https://genforce.github.io/mganprior/)).

SIGGRAPH'19: **Semantic Photo Manipulation with a Generative Image Prior**. [paper](http://ganpaint.io/)<br>
*comment*: it considers the GAN model as a image prior, then fine-tunes the weights of the pretrained network to invert a given image, finally use the unit semantics discovered by the following GAN dissection to manipulate the image content. 

ICLR'19: **GAN Dissection: Visualizing and Understanding Generative Adversarial Networks**. [paper](http://gandissect.csail.mit.edu/)<br>
*comment*: one of the earliest works looked into the interpretability of generative models PG-GAN.

## Image to Image (Pix2pix and cycleGAN types)

ECCV2020: **Contrastive Learning for Unpaired Image-to-Image Translation**. [paper](https://arxiv.org/pdf/2007.15651.pdf)<br>
*comment*: Use local contrastic loss to improve quality of pix2pix architecture.

ECCV2020: **Learning to Factorize and Relight a City**. [paper](https://arxiv.org/pdf/2008.02796)<br>
*comment*: use pix2pix arch + code swap to factorize the intrinsic images of street-view scenes. It looks similar to the following work.

arXiv: **Swapping Autoencoder for Deep Image Manipulation**. [paper](https://arxiv.org/pdf/2007.00653.pdf)<br>
*comment*: use code swapping to disentangle textures and contents. 

a recent survey dated Aug 2020: **Generative Adversarial Networks for Image and Video Synthesis: Algorithms and Applications**. [paper](https://arxiv.org/pdf/2008.02793.pdf)

## Generative models for 3D

ECCV2020: **Learning Gradient Fields for Shape Generation**. [paper](https://arxiv.org/abs/2008.06520) [page](https://www.cs.cornell.edu/~ruojin/ShapeGF/)<br>
*comment*: 3D point cloud generation by a mixture of Gaussian model.

ICCV2019: **PointFlow : 3D Point Cloud Generation with Continuous Normalizing Flows**. [paper](https://arxiv.org/abs/1906.12320) [page](https://www.guandaoyang.com/PointFlow/)<br>
*comment*: invertible Normalizing flow model for 3D point generation.

SIGGRAPH ASIA 2020: **Scene Mover: Automatic Move Planning for Scene Arrangement by Deep Reinforcement Learning**. [paper]() [page](https://reposhub.com/python/deep-learning/HanqingWangAI-SceneMover.html)

## Generative models for drug discovery

Nature MI: **Generative molecular design in low data regimes**. [paper](https://www.nature.com/articles/s42256-020-0160-y.epdf?author_access_token=kx71VwOu26XWGELCg3BP-NRgN0jAjWel9jnR3ZoTv0MojvyIaQWNqzF7aemIUbYlNUc8tqoGgWco3JoR6d8H9plcxmpko09VfAUvw6-sCHyp8bABy7FhZ89AUc_da9ZU3s4YWQy4gK0meFq2XLhHYA%3D%3D) [code](https://github.com/ETHmodlab/virtual_libraries)

Nature MI: **Direct steering of de novo molecular generation with descriptor conditional recurrent neural networks**. [paper](https://www.nature.com/articles/s42256-020-0174-5) 

Medium: **Creating Molecules from Scratch I: Drug Discovery with Generative Adversarial Networks** [link](https://medium.com/neuromation-blog/creating-molecules-from-scratch-i-drug-discovery-with-generative-adversarial-networks-9d42cc496fc6)

## Generative models for animation and locomotion

SIGGRAPH20: **CARL: Controllable Agent with Reinforcement Learning for Quadruped Locomotion**. [paper](https://inventec-ai-center.github.io/projects/CARL/CARL.pdf) [page](https://inventec-ai-center.github.io/projects/CARL/index.html) <br>
*comment*: a physics-based controller is composed of three stages. 1)imitation learning for low-level control that specifies the agent's movement at the joint level. 2)GAN control adapter to approximate the natural action distribution in the high-level user control. 3)DRL finetuning to improve the controller's ability to adapt to unseen scenarios. High-level user controls over speed and heading. The idea of using high-level GAN controller in second stage to mimic the behavior of the low-level
trained controller, achieved by a GAN loss, is clever. There are the paired labels c_high and c_low.  Controllable agent is the way to go!

SIGGRAPH20: **Local Motion Phases for Learning Multi-Contact Character Movements**. [paper](http://www.ipab.inf.ed.ac.uk/cgvu/basketball.pdf) [page](http://www.starke-consult.de/portfolio/assets/content/work/14/page.html) <br>
*comment*: A generative control model is introduced to produce a variation of realistic movements from the coarse user control signal, which is an encoder-decoder structure + GAN loss. 

SIGGRAPH20: **Character Controllers using Motion VAEs**. [paper](https://www.cs.ubc.ca/~van/papers/2020-TOG-MVAE/2020-TOG-MVAE.pdf) [page](https://www.cs.ubc.ca/~hyuling/projects/mvae/) <br>
*comment*: A clear two-stage framework to train character controllers. 1) Train a motion synthesis VAE model: given p_t-1 and p_t for encoder, decoder output p_t. 2) Throw away encoder, train a controller which outputs latent code to the decoder. 

SIGGRAPH20: **Unpaired Motion Style Transfer from Video to Animation**.[paper](https://uploads-ssl.webflow.com/51e0d73d83d06baa7a00000f/5cab99df4998decfbf9e218e_paper-01.png) [page](https://deepmotionediting.github.io/style_transfer) <br>
*comment*: Disentangling content code and style code for motion style transfer. 

AI4Animation: **AI4Animation: Deep Learning, Character Animation, Control**.[link](https://github.com/sebastianstarke/AI4Animation). <br> 
*comment*: several relevant SIGGRAPH papers and code and locomotion data.

## Relevant Researchers (random order)

[Craig Yu](https://craigyuyu.github.io/home/research.html): faculty at GMU. on graphics

[Junyan Zhu](https://www.cs.cmu.edu/~junyanz/): Adobe researcher and faculty at CMU. on vision + graphics

[Tero Karras](https://scholar.google.fi/citations?user=-50qJW8AAAAJ&hl=en): nvidia research, leading author of stylegan, as Kaiming He in the field of generative modeling. oh man, i love this guy's work.

[Ming-Yu Liu](http://mingyuliu.net/): nvidia researcher on computer vision

[David Bau](https://people.csail.mit.edu/davidbau/home/): phd at mit, my collaborator. David is a guru of graphic interface! on model understanding and interpretability.

[Alexei Efros](https://people.eecs.berkeley.edu/~efros/): faculty at berkeley. without doubt, the pixel god-father!

[Aaron Hertzman](https://research.adobe.com/person/aaron-hertzmann/): Principal scientist at Adobe. without doubt, the pioneer in image generation.
