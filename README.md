# Papers on generative modeling 

### Well, check out our GenForce work first!

[GenForce](https://genforce.github.io): may generative force be with you. Refer to this page for our latest work.

CVPR2020: Interpreting the Latent Space of GANs for Semantic Face Editing. [paper](https://genforce.github.io/interfacegan)
*comment*: use facial classifiers to discover the interpretable dimensions emerged in the GANs trained to synthesize faces.

CVPR2020: Image Processing Using Multi-Code GAN Prior. [paper](https://genforce.github.io/mganprior/)
*comment*: use the pretrained GAN model as a representation prior to facilitate a series of image processing tasks, such as colorization, super-resolution, denoising.

ECCV2020: In-Domain GAN Inversion for Real Image Editing. [paper](https://genforce.github.io/idinvert/)
*comment*: Inverted code from Image2StyleGAN does not have enough manipulatability. This work proposes to use an encoder to regularize the optimization to preserve the manipulatability. A novel semantic difusion application is also proposed. 

arXiv: Closed-Form Factorization of Latent Semantics in GANs. [paper](https://genforce.github.io/sefa/)
*comment*: Unsupervised discovery of the interpretable dimensions in learned GAN models. The algorithm works blazingly fast within only 1 second!

arXiv: Generative Hierarchical Features from Synthesizing Images. [paper](https://genforce.github.io/ghfeat/)
*comment*: It considers the pretrained StyleGAN model as a learned loss (similar to perceptual loss using learned VGG) to train a hierarchical encoder. This work calls to explore various applications of the encoder for both for discriminative tasks and generative tasks. It also echoes my opinion that ImageNet classification is NOT the only way to evaluate the merits of learned features from self-supervised learning. There are so many visual tasks out there, why just stick to dog
classification on ImageNet?? (fun fact, there are about 150 dog classes out of the 1000 ImageNet classes).

### Latent code to Image (StyleGAN and BigGAN types)

ECCV2020: StyleGAN2 Distillation for Feed-forward Image Manipulation. [paper](https://arxiv.org/pdf/2003.03581.pdf) [Page](https://github.com/EvgenyKashin/stylegan2-distillation).
*comment*: use InterfaceGAN pipeline to generate paired data from pretrained styleGAN, then use the paired data to train a pix2pix network. 

ECCV2020: Rewriting a Deep Generative Model. [paper](https://rewriting.csail.mit.edu/).
*comment*: interactively replace the unit semantic patterns. David is the guru of interface, always. 

ECCV2020: Exploiting Deep Generative Prior for Versatile Image Restoration and Manipulation. [paper]([Project page])
*comment*: similar to the following SIGGRAPH'19 work, it develops an inversion method for retraining the pretrained model to invert an image, then applies to a series of image processing tasks (similar to [mGAN prior](https://genforce.github.io/mganprior/)).

SIGGRAPH'19: Semantic Photo Manipulation with a Generative Image Prior. [paper](http://ganpaint.io/)
*comment*: it considers the GAN model as a image prior, then fine-tunes the weights of the pretrained network to invert a given image, finally use the unit semantics discovered by the following GAN dissection to manipulate the image content. 

ICLR'19: GAN Dissection: Visualizing and Understanding Generative Adversarial Networks. [paper](http://gandissect.csail.mit.edu/).
*comment*: one of the earliest works looked into the interpretability of generative models PG-GAN.

### Image to Image (Pix2pix and cycleGAN types)

ECCV2020: Contrastive Learning for Unpaired Image-to-Image Translation. [paper](https://arxiv.org/pdf/2007.15651.pdf).
*comment*: Use local contrastic loss to improve quality of pix2pix architecture.

ECCV2020: Learning to Factorize and Relight a City. [paper](https://arxiv.org/pdf/2008.02796).
*comment*: use pix2pix arch + code swap trick to disentangle the variation factor of street-view images. seems similar to the following work.

arXiv: Swapping Autoencoder for Deep Image Manipulation. [paper](https://arxiv.org/pdf/2007.00653.pdf).
*comment*: use code swapping to disentangle textures and contents. 

a recent survey on GAN dated 6 Aug 2020: Generative Adversarial Networks for Image and Video Synthesis: Algorithms and Applications. [paper](https://arxiv.org/pdf/2008.02793.pdf)

### Generative models for 3D

### Generative models beyond image

SIGGRAPH ASIA 2020: Scene Mover: Automatic Move Planning for Scene Arrangement by Deep Reinforcement Learning. [paper]() [page](https://reposhub.com/python/deep-learning/HanqingWangAI-SceneMover.html)

### Relevant Researchers (random order)

[Craig Yu](https://craigyuyu.github.io/home/research.html): faculty at GMU. on graphics

[Junyan Zhu](https://www.cs.cmu.edu/~junyanz/): Adobe researcher and faculty at CMU. on vision + graphics

[Tero Karras](https://scholar.google.fi/citations?user=-50qJW8AAAAJ&hl=en): nvidia research, leading author of stylegan, as Kaiming He in the field of generative modeling. oh man, i love this guy's work. 

[Ming-Yu Liu](http://mingyuliu.net/): nvidia researcher on computer vision

[David Bau](https://people.csail.mit.edu/davidbau/home/): phd at mit, my collaborator. David is a guru of graphic interface! on model understanding and interpretability.

[Alexei Efros](https://people.eecs.berkeley.edu/~efros/): faculty at berkeley. without doubt, the pixel god-father!

[Aaron Hertzman](https://research.adobe.com/person/aaron-hertzmann/): Principal scientist at Adobe. without doubt, the pioneer in image generation. 
