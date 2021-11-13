# Reset Networks : Towards topography at scale
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-iUCNMw8Ry-y4PF0xu_jGFpx0ghjTp4i?usp=sharing)

## What is a reset network?
A reset network is a composition of several levels of neural networks (typically CNNs), in which the outputs of many networks at one level are reshaped into a spatial map which serves as input for the next level.

![reset_networks_gen](https://user-images.githubusercontent.com/13241166/140661564-94a53cde-32c2-4b81-b4b6-db2c2fcb58fa.png)

The general form of a Reset network is shown in the figure above. It has an arbitrary depth of levels, each consisting of several networks operating in parallel on the same input.

Reset networks include in particular the following family of depth 2, where level 1 is obtained by reshaping and concatenating the outputs of nxn parallel networks into a single map, called the grid hereafter, which then serves as input for a final network:

![reset_networks_2](https://user-images.githubusercontent.com/13241166/140658409-e557f449-8af9-46a2-a405-6c62fc45687d.png)

The master network forces the grid of subnetworks below to organize in order to solve the task, distributing work in a way that creates topography.

The notebooks hosted in this Github and on Google Colab demonstrate that Reset networks can perform classification at scale -which arguably is not so suprising- while also exhibiting emergent topographic organization at each level. We further present evidence that this topography is of the kind needed to make sense of certain phenomena in the visual cortex of mammals.

## Why are they relevant to cortical topography?

Cortical topography in the strict sense is the notion that "nearby neurons in the cortex have receptive fields at nearby locations in the world" [[1]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4074559/). When understood as applying also to local fields or voxels as well as to neurons, this is a widespread phenomenon in brain imaging, observed throughout the visual cortex as well as in some associative areas. 

### Topography for numbers in parietal cortex

In parietal cortex, voxels selective for similar numbers are more likely to be contiguous, a phenomenon which has only been partially explained as a planar diffusion process of number codes, due to an underlying locally and randomly connected network of cortical units [2]. This network, however, did not process real stimuli. As the figure below shows, a Reset Network with a single 8x8 grid, can be trained to map images of numbers onto number codes, and succeeds in reproducing topographic organization.

![Reset_numerotopy](https://user-images.githubusercontent.com/13241166/141204652-fa07b2f1-b3dd-4043-98e3-dd7da3caf01c.png)

Number topography is clearly visible on the map of number preferences, and is quantified in the middle plot above, where it can also be seen to emerge quickly during training. Topography and neighborhood similarity (right plot) are both quite significantly above the levels obtained for shuffled selectivity maps. Also notable is the tendency of subnetworks to specialize for specific numbers, or in the same ballpark.

### Topography for categorical areas in ventral occipitotemporal cortex

In ventral occipitotemporal cortex, more than two decades of studies have established the presence of areas selective for various widespread visual categories, in particular faces, bodies, tools, houses, and words. While there is no shortage of computational models able to reproduce many caracteristics of the visual system, including some of vOTC, only one [3] achieves both topography and scale at the same time - with topography being problematic as it emerges from two different notions of space. Reset networks achieve topography at scale in a conceptually simple way.

![Reset network for vOTC](https://user-images.githubusercontent.com/13241166/141536355-621178f4-555b-4863-8639-be40cb61c21c.png)

The left pannel in the figure shows a Reset network classifier trained on Cifar-100. The right pannel shows category preferences on the grid after training. Only 3 categories are considered - objects, houses and people - which were obtained by agregating the relevant cifar-100 classes. Clustering is visible in the map, and quantified in the subplots above (although with different indicators as before for numerotopy).

### VOTC topography and the Visual Word Form Area

A closely related topic is that of the so-called Visual Word Form Area, which, with the benefit of insight and despite its discoverers best intent upon naming it, is neither visual (congenital blind subjects have it too), word specific (it is also active for individual letters), nor a single area (it appears to be organized in patches). But names have great inertia, and this one does convey well the idea of a localized region selective for stimuli related to words. There is currently no model of the VWFA that can account for its specific place within the topography of vOTC. The following figure describes what a reset network of the VWFA could look like.

![reset_vwfa](https://user-images.githubusercontent.com/13241166/141659718-402bf570-4242-4f5b-a6ab-2109511729ea.png)





## References
[1] Patel GH, Kaplan DM, Snyder LH. Topographic organization in the brain: searching for general principles. Trends Cogn Sci. 2014;18(7):351-363. doi:10.1016/j.tics.2014.03.008

[2] Hannagan T, Nieder A, Viswanathan P, Dehaene S. A random-matrix theory of the number sense. Phil. Trans. R. Soc. B. 2018;373:20170253.
http://doi.org/10.1098/rstb.2017.0253

[3] Lee H, Margalit E, Jozwik KM, Cohen MA, Kanwisher N, Yamins DL, DiCarlo JJ. (2020). Topographic deep artificial neural networks reproduce the 
hallmarks of the primate inferior temporal cortex face processing network. ![bioRxiv](https://www.biorxiv.org/content/10.1101/2020.07.09.185116v1.full.pdf). 
