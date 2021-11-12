# Reset Networks : Towards topography at scale
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-iUCNMw8Ry-y4PF0xu_jGFpx0ghjTp4i?usp=sharing)

## What is a reset network?
A reset network is a composition of neural networks in which non-spatial channels are reset into a spatial map, at least once during the course of computations.

![reset_networks_gen](https://user-images.githubusercontent.com/13241166/140661564-94a53cde-32c2-4b81-b4b6-db2c2fcb58fa.png)

The general form of a Reset network is shown in the figure above. It has an arbitrary depth of levels, each level consisting of a number of networks operating in parallel on the same input.

Reset networks include in particular the following family of depth 2, where level 1 is obtained by reshaping and concatenating the outputs of nxn parallel networks into a single map, which serves as input for a unique -master- network:

![reset_networks_2](https://user-images.githubusercontent.com/13241166/140658409-e557f449-8af9-46a2-a405-6c62fc45687d.png)

The notebooks hosted in this Github and on Google Colab demonstrate that Reset networks can perform classification at scale -which arguably is not so suprising- while also exhibiting emergent topographic organization at each level. We further present evidence that this topography is of the kind needed to make sense of certain phenomena in the visual cortex of mammals.

## Why are they relevant to cortical topography?

Cortical topography in the strict sense is the notion that "nearby neurons in the cortex have receptive fields at nearby locations in the world" [[1]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4074559/). When understood as applying also to local fields or voxels as well as to neurons, this is a widespread phenomenon in brain imaging, observed throughout the visual cortex as well as in some associative areas. 

### Topography for numbers in parietal cortex

In parietal cortex, voxels selective for similar numbers are more likely to be contiguous, a phenomenon which has only been partially explained as a planar diffusion process of number codes, due to an underlying locally and randomly connected network of cortical units [2]. This network, however, did not process real stimuli. As the figure below shows, a Reset Network trained to map images of numbers onto number codes succeeds in reproducing topographic organization.

![Reset_numerotopy](https://user-images.githubusercontent.com/13241166/141204652-fa07b2f1-b3dd-4043-98e3-dd7da3caf01c.png)

Number topgraphy is clearly visible on the selectivity map, and quantified in the plot above, where it can also be seen to emerge very quickly during training. Topography and neighborhood are both very significantly above levels obtained for shuffled selectivity maps. Also notable is the tendency of subnetworks to specialize for specific numbers, + or -1.

### Topography in ventral occipito-temporal cortex

## References
[1] Patel GH, Kaplan DM, Snyder LH. Topographic organization in the brain: searching for general principles. Trends Cogn Sci. 2014;18(7):351-363. doi:10.1016/j.tics.2014.03.008

[2] Hannagan T, Nieder A, Viswanathan P, Dehaene S. A random-matrix theory of the number sense. Phil. Trans. R. Soc. B. 2018;373:20170253.
http://doi.org/10.1098/rstb.2017.0253
