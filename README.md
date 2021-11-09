# Reset Networks : Towards topography at scale
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-iUCNMw8Ry-y4PF0xu_jGFpx0ghjTp4i?usp=sharing)

## What is a reset network?
A reset network is a composition of neural networks in which non-spatial channels are reset into a spatial map, at least once during the course of computations.

![reset_networks_gen](https://user-images.githubusercontent.com/13241166/140661564-94a53cde-32c2-4b81-b4b6-db2c2fcb58fa.png)

The general form of a Reset network is shown in the figure above. It has an arbitrary depth of levels, each level consisting of a number of networks operating in parallel.

Reset networks include in particular the following family of depth 2, where level 1 is obtained by reshaping and concatenating the outputs of nxn parallel networks into a single map, which serves as input for a unique -master- network:

![reset_networks_2](https://user-images.githubusercontent.com/13241166/140658409-e557f449-8af9-46a2-a405-6c62fc45687d.png)

The notebooks hosted in this Github and on Google Colab demonstrate that Reset networks can perform classification at scale -which arguably is not so suprising- while also exhibiting emergent topographic organization at each level. We further present evidence that this topography is of the kind needed to make sense of certain phenomena in the visual cortex of mammals.
