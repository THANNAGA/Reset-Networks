# Reset Networks : rethinking the link between CNNs and the visual cortex
Hannagan T. Reset Networks: Emergent Topography in Networks of Convolutional Neural Networks. [bioRxiv 2021.11.19.469308](https://www.biorxiv.org/content/10.1101/2021.11.19.469308v2)

## What is a Reset network?
A Reset network is a composition of several neural networks - typically several levels of CNNs - where outputs at one level are reshaped into a spatial input for the next level. The seed for this idea came from multiple discussions with [Thibault Fouqueray](https://www.linkedin.com/in/thibault-fouqueray/?originalSubdomain=fr) (Stellantis), on how to implement a neural space where networks performing similar tasks would end-up being neighbors, and I am also grateful to [Florence Bouhali](https://scholar.google.fr/citations?user=0J6-PIsAAAAJ&hl=en) (UCSF) for inputs and encouragements in the development of the approach.

<img src="https://user-images.githubusercontent.com/13241166/140661564-94a53cde-32c2-4b81-b4b6-db2c2fcb58fa.png" width="500" height="700" />

The general form of a Reset network is shown in the figure above. It has an arbitrary depth of levels, each consisting of several networks operating in parallel on the same input.

Reset networks include in particular the following family of depth 2, where level 1 is obtained by reshaping and concatenating the outputs of nxn parallel networks into a single map, called "grid" hereafter, which then serves as input for a final network. We will refer to this family of networks as Reset(n).

![reset_networks_2](https://user-images.githubusercontent.com/13241166/140658409-e557f449-8af9-46a2-a405-6c62fc45687d.png)

The master network forces the grid of subnetworks underneath to organize in order to solve the task, distributing work in a way that creates topography.

The notebooks hosted in this Github and on Google Colab demonstrate that Reset networks can perform classification at scale -which arguably is not so suprising- while also exhibiting emergent topographic organization at each level. 

## Why are Reset networks relevant to cortical topography?

Cortical topography in the strict sense is the notion that "nearby neurons in the cortex have receptive fields at nearby locations in the world" [1]. When understood as applying also to local fields or voxels as well as to neurons, this is a widespread phenomenon in brain imaging, observed throughout the visual cortex as well as in some associative areas. 

### Topography for numbers in parietal cortex [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-iUCNMw8Ry-y4PF0xu_jGFpx0ghjTp4i?usp=sharing)

In parietal cortex, voxels selective for similar numbers are more likely to be contiguous, a phenomenon which has only been partially explained as a planar diffusion process of number codes, using a random matrix model which was not designed to process real stimuli [2]. As the figure below shows, a Reset(8) Network with a single 8x8 grid,  can be trained to map images of numbers onto number codes, and succeeds in reproducing topographic organization. We emphasize that we would expect the same topography to emerge for classification of dots, or any kind of stimuli with countable objects.

![Reset_numerotopy](https://user-images.githubusercontent.com/13241166/141204652-fa07b2f1-b3dd-4043-98e3-dd7da3caf01c.png)

Number topography is visible on the map of number preferences in this figure, and is quantified in the middle plot above, where it can also be seen to emerge quickly during training. Our two indices, topography and neighborhood similarity (resp. middle and right plot) are both quite significantly above what they are for a shuffled selectivity map. Also notable is the tendency of subnetworks to specialize for specific numbers, or numbers in the same ballpark: check out in this respect the videos for ![Reset(2)](https://github.com/THANNAGA/Reset-Networks/blob/main/Topography%20for%20numbers/history_monitor_ResetNumber2_Numbers10.mp4) and ![Reset(4)](https://github.com/THANNAGA/Reset-Networks/blob/main/Topography%20for%20numbers/history_monitor_ResetNumber4_Numbers10.mp4). Pretrained Reset networks are available in the folder ![Topography for numbers](https://github.com/THANNAGA/Reset-Networks/tree/main/Topography%20for%20numbers), where one can also find the above mentioned movies, showing how number topography evolves during training of these networks.

### Topography for categorical areas in ventral occipitotemporal cortex [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RhRNCYmUEr1lppWrtrmoJaPbCv9XQR4X)

In ventral occipitotemporal cortex, more than two decades of studies have established the presence of areas selective for various widespread visual categories, in particular faces, bodies, tools, houses, and words. While there is no shortage of computational models able to reproduce many caracteristics of the visual system, including some of vOTC, only one [3] achieves both topography and scale at the same time - with topography being problematic as it requires two different notions of space to coexist (thanks are due to Hyodong Lee for exchanges that provided useful information on Topographic Deep Artificial Neural Networks). By contrast, the way Reset networks achieve topography at scale is conceptually straightforward.

![Reset network for vOTC](https://user-images.githubusercontent.com/13241166/141536355-621178f4-555b-4863-8639-be40cb61c21c.png)

The left pannel in the figure presents a Reset network classifier trained on Cifar-100. The right pannel shows category preferences on the grid after training. Only 3 categories are considered - objects, houses and people - which were obtained by agregating the relevant cifar-100 classes. Clustering is visible in the map, and quantified in the subplots above, although with slightly different indicators as before for numerotopy: we here use a simple thresholded d-prime to determine category preference (upper middle plot), and provide a density index (upper right plot) corresponding to the proportion of total activity on the map that falls within a given category.
Again, pretrained Reset networks are available in the folder ![Topography for cifar](https://github.com/THANNAGA/Reset-Networks/tree/main/Topography%20for%20cifar), as well as movies showing how topography evolves during training of these networks, for the 3 categories considered.

### VOTC topography and the Visual Word Form Area - with ![Florence Bouhali](https://scholar.google.fr/citations?user=0J6-PIsAAAAJ&hl=en)

A closely related topic is that of the so-called Visual Word Form Area --which, with the benefit of insight and despite its discoverers' best intent upon naming it, is neither visual (congenital blind subjects have it too), word-level specific (it is also active for individual letters), nor probably a single area (it appears to be organized in patches). But names have great inertia, and this one does convey well the idea of a very narrow cortical region selective for stimuli related to words, as opposed to e.g. objects or number symbols. While some efforts have gone into modeling the VWFA [4], currently no model can account for its specific place within the topography of vOTC. The following network, designed with ![Florence Bouhali](https://scholar.google.fr/citations?user=0J6-PIsAAAAJ&hl=en), describes what a Reset network of the VWFA could look like.

<img src="https://user-images.githubusercontent.com/13241166/141679793-4c477a7f-3f69-498c-b634-4ba9d9be1ab1.png" width="500" height="700" />

First, this Reset network would have 2 intermediate grids, P and A, standing for the posterior and anterior axis in vOTC. This is not an innovation, but now Reset networks allow for something interesting to happen. In addition to the posterior-to-anterior gradient, we can capture a lateral-to-medial gradient by ensuring that networks in the P grid see different parts of the input depending on where they are: left-located (lateral) networks on the P grid would receive input from the center of the image, whereas right-located (medial) networks would receive input from the periphery. In other words, we build into the model a lateral-to-medial gradient in vOTC by exploiting its well-documented correspondence with center/periphery processing [5]. We insist that such a relation cannot easily be built into a CNN, because of location invariance.

## Discussion

### Classification performance
We have shown that Reset networks can classify standard computer vision datasets such as CIFAR. However and as the figure below shows, at this stage their performance is disappointing, only at best matching that of a single Resnet 20, while having many more parameters. 

![Reset performance Cifar10Cifar100](https://user-images.githubusercontent.com/13241166/143680476-ff8fd5eb-abce-40aa-8da9-a92edf0b0ed8.png)

One reason could be that in our simulations, spatial resets between levels were always done by reshaping the subnetworks' outputs, which constitute an information botteneck. Reshaping prior to the subnetwork's output, e.g. the dense layer or before, might be a more astute choice. We also observe that the full resources of the Reset network don't seem to be used: some subnetwork units are more active than others. This can be alleviated to some extent by using dropout, or another kind of regularization on the grid.

### Regularization by auto-encoding
In the course of our investigations (not shown here), we observed that Reset networks that were based on smaller subnetworks than Reset20, performed much better when the second level had 2 networks: one that classified the input, and another that tried to reconstruct the input from the grid. Auto-encoding in this situation appears to act as an efficient regularizer for classification, forcing the error gradient to be distributed across the whole grid rather than to be drawn by one, or just a few subnetworks. Such regularization effects of auto-encoding have been reported before for standard classifiers [6].

### Topography
Reset networks constitute a novel mechanism for topography to emerge in deep learning. We have presented solid evidence that they can reproduce at least two examples of topographic organization: in parietal cortex for numbers, and in ventral Occipitotemporal cortex for the so-called "categorical areas". A related point is that Reset networks provide a way to implement a cortical gradient, the mapping between foveal/peripheral input and lateral/medial in visual cortex, which is not easily captured within the standard assumptions of CNNs.

### Adding networks when necessary: the width and depth of Reset networks
Reset networks align well with a view of neural development in which, as an alternative to recycling extant neural material, neural resources can also be recruited in the system if needed. Learning a new task could thus require only to widen the system by adding a network at the current level, with different networks possibly trained on different tasks. If expertise from previously learned tasks is required, the system can be made deeper by reshaping network outputs at the current level and creating a new level.

## Conclusion
Reset networks show that topography must emerge in deep CNN classifiers, when composed with one another. In this view, the topographic cortex should not be modeled as a single classifier, however deep and richly organized, but as a sequence of levels of neural network classifiers. This rests on the idea that the cortex has the ability to compose networks with one another if need be, and predicts that the outputs, or the late computational stages, of cortical classifiers are either spatially organized, or somehow reshaped spatially during the course of composition.

## Citation
Hannagan T. Reset networks: Topography at scale. [bioRxiv 2021.11.19.469308](https://www.biorxiv.org/content/10.1101/2021.11.19.469308v2)


## References
[1] Patel GH, Kaplan DM, Snyder LH. Topographic organization in the brain: searching for general principles. Trends Cogn Sci. 2014;18(7):351-363. [doi:10.1016/j.tics.2014.03.008](https://pubmed.ncbi.nlm.nih.gov/24862252/)

[2] Hannagan T, Nieder A, Viswanathan P, Dehaene S. A random-matrix theory of the number sense. Phil. Trans. R. Soc. B. 2018;373:20170253.
[doi:10.1098/rstb.2017.0253](https://royalsocietypublishing.org/doi/10.1098/rstb.2017.0253)

[3] Lee H, Margalit E, Jozwik KM, Cohen MA, Kanwisher N, Yamins DL, DiCarlo JJ. Topographic deep artificial neural networks reproduce the 
hallmarks of the primate inferior temporal cortex face processing network. 2020 [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.07.09.185116v1.full.pdf). 

[4] Hannagan T, Agrawal A., Cohen L, Dehaene S. Emergence of a compositional neural code for written words: Recycling of a convolutional neural network for reading. Proceedings of the National Academy of Sciences Nov 2021, 118 (46) e2104779118; [doi: 10.1073/pnas.2104779118](https://www.pnas.org/content/118/46/e2104779118)

[5] Op de Beeck HP, Pillet I, Ritchie JB. Factors Determining Where Category-Selective Areas Emerge in Visual Cortex. Trends Cogn Sci. 2019 Sep;23(9):784-797.[doi: 10.1016/j.tics.2019.06.006.](https://pubmed.ncbi.nlm.nih.gov/31327671/)

[6] Le L, Patterson A, White M. Supervised autoencoders: Improving generalization performance with unsupervised regularizers. In [Advances in Neural Information Processing Systems. 2018. 107–117](https://proceedings.neurips.cc/paper/2018/file/2a38a4a9316c49e5a833517c45d31070-Paper.pdf).

