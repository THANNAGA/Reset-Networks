# Reset Networks : rethinking the link between CNNs and the visual cortex
Hannagan T. Reset Networks: Emergent Topography by Composition of Convolutional Neural Networks. [bioRxiv 2021.11.19.469308](https://www.biorxiv.org/content/10.1101/2021.11.19.469308v3)

## TL-DR
If you stack CNN classifiers on top of one another, they will self-organize topographically.

## What is a Reset network?
A Reset network is a composition of several neural networks - typically several levels of CNNs - where outputs at one level are reshaped into a spatial input for the next level. The seed for this idea came from multiple discussions with [Thibault Fouqueray](https://www.linkedin.com/in/thibault-fouqueray/?originalSubdomain=fr) (Stellantis), on how to implement a neural space where networks performing similar tasks would end-up being neighbors, and I am also grateful to [Florence Bouhali](https://scholar.google.fr/citations?user=0J6-PIsAAAAJ&hl=en) (UCSF) for inputs and encouragements in the development of the approach.

<img src="https://user-images.githubusercontent.com/13241166/140661564-94a53cde-32c2-4b81-b4b6-db2c2fcb58fa.png" width="500" height="700" />

The general form of a Reset network is shown in the figure above. It has an arbitrary depth of levels, each consisting of several networks operating in parallel on the same input.

Reset networks include in particular the following family of depth 2, where level 1 is obtained by reshaping and concatenating the outputs of nxn parallel networks into a single map, called "grid" hereafter, which then serves as input for a final network. We will refer to this family of networks as Reset(n).

![reset_networks_2](https://user-images.githubusercontent.com/13241166/140658409-e557f449-8af9-46a2-a405-6c62fc45687d.png)

The master network forces the grid of subnetworks underneath to organize in order to solve the task, distributing work in a way that creates topography. Exactly how this happens is still not clear, but the results below provide a few hints.

The notebooks hosted in this Github and on Google Colab demonstrate that Reset networks can perform classification at scale -which arguably is not so suprising- while also exhibiting emergent topographic organization. 

## Reset networks show clustering for MNIST, Fashion MNIST and CIFAR-10

Let's start by training 3 Reset(8) networks, each on a classic computer vision datasets: MNIST, Fashion MNIST and CIFAR-10. In each case, the networks reached standard performance levels on the test sets, but more interestingly, the following figure shows the model's grid after 20 epochs of training.

![topography_across_domains](https://user-images.githubusercontent.com/13241166/145676025-584145c8-2980-4944-a85f-14d6e4d6872f.png)

The upper plots present converged preference maps -the class preference of each unit on the 32x32 grid of the trained model- whereas the lower plots quantify the amount of clustering on each map, at each point during training. 

A unit's preference is given by the highest d-prime, over each target class, of the unit's responses to this target class against all other classes. If this maximum d-prime is below an arbitrary threshold, here set to 2, the unit is said to have no preference (white units on the preference map). Clustering is then defined as the average size, over all target classes, of the connected components present on the map for this class. More details can be found in the [paper](https://www.biorxiv.org/content/10.1101/2021.11.19.469308v3).

There is clustering for each of the three domains considered, with some variations across domains -for instance CIFAR-10 produces more clustering, while Fashion MNIST produces fewer category selective units.

Clustering also happens quite quickly: it is largely in place after the first training epoch.

## Why are Reset networks relevant to cortical topography?

Cortical topography in the strict sense is the notion that "nearby neurons in the cortex have receptive fields at nearby locations in the world" [1]. However the term has come to take a wider meaning: it is often understood as applying also to local fields or voxels as well as to neurons, and to refer to any kind of selectivity, not just location selectivity. In this wider sense, topography is a widespread phenomenon in brain imaging, observed throughout the visual cortex as well as in some associative areas. 

### Topography for numbers in parietal cortex [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-iUCNMw8Ry-y4PF0xu_jGFpx0ghjTp4i?usp=sharing)

In parietal cortex, voxels selective for similar numbers are more likely to be contiguous, a phenomenon which is not yet well understood. Of course, we have just seen that Reset networks will self-organize when trained to classify hand-written digits, so one might argue that this is a good start for an explanation. However, classifying MNIST is not entirely satisfying a task from a neuroscientific point of view: it is very likely that kids map written digits not onto one-hot labels, but onto pre-existing (possibly innate!) number representations that have a rather special format. 

The nature of these codes has been thoroughly studied in [2]: a lot of phenomena could be explained if number codes were sparsely distributed vectors, with exponentially more overlap between successive number codes as numbers increase. Therefore, it would be more convincing if Reset networks could reproduce number topography by mapping digit images onto these realistic number codes (and even more so if the task was to map visual scenes with n objects to the number code for n, but that's yet another project).

As the figure below shows, a Reset(8) Network (with a single 8x8 grid) can be trained to map images of digits onto number codes, and succeeds in reproducing topographic organization. We emphasize that we would expect the same topography to emerge for classification of dots, or any kind of stimuli with countable objects.

![Figure results number codes](https://user-images.githubusercontent.com/13241166/145191103-c99ca3d9-11c9-4f72-a0b1-9221342c4afe.png)

Number topography is visible on the map of number preferences in this figure, and is quantified in the middle plot above, where it can also be seen to emerge quickly during training. Our two indices, cluster size and neighborhood similarity (resp. middle and right plot) are both quite significantly above what they are for a shuffled selectivity map. Also notable is the tendency of subnetworks to specialize for specific numbers, or numbers in the same ballpark: check out in this respect the videos for ![Reset(2)](https://github.com/THANNAGA/Reset-Networks/blob/main/Topography%20for%20numbers/history_monitor_ResetNumber2_Numbers10.mp4) and ![Reset(4)](https://github.com/THANNAGA/Reset-Networks/blob/main/Topography%20for%20numbers/history_monitor_ResetNumber4_Numbers10.mp4). Pretrained Reset networks are available in the folder ![Topography for numbers](https://github.com/THANNAGA/Reset-Networks/tree/main/Topography%20for%20numbers), where one can also find the above mentioned movies, showing how number topography evolves during training.

### Topography for categorical areas in ventral occipitotemporal cortex [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RhRNCYmUEr1lppWrtrmoJaPbCv9XQR4X)

In ventral occipitotemporal cortex, more than two decades of studies have established the presence of areas selective for various widespread visual categories, in particular faces, bodies, tools, houses, and words. While there is no shortage of computational models able to reproduce many caracteristics of the visual system, including some of vOTC, only one [3] achieves both topography and scale at the same time - with topography being problematic as it requires two different notions of space to coexist. By contrast, the way Reset networks achieve topography at scale is conceptually straightforward. 
I thank ![Hyodong Lee](https://scholar.google.com/citations?user=1QvDhAQAAAAJ&hl=en), who would likely disagree with my assessment of the problem, for email exchanges on her innovative work on Topographic Deep Artificial Neural Networks. 

![Reset network for vOTC](https://user-images.githubusercontent.com/13241166/141536355-621178f4-555b-4863-8639-be40cb61c21c.png)

The left pannel in the figure presents a Reset network classifier trained on Cifar-100. The right pannel shows category preferences on the grid after training. Only 3 categories are considered - objects, houses and people - which were obtained by agregating the relevant cifar-100 classes. Clustering is visible on the map, and quantified in the subplots above, although with slightly different indicators as before for numerotopy: we here use a simple thresholded d-prime to determine category preference (upper middle plot), and provide a density index (upper right plot) corresponding to the proportion of total activity on the map that falls within a given category.

Again, pretrained Reset networks are available in the folder ![Topography for cifar](https://github.com/THANNAGA/Reset-Networks/tree/main/Topography%20for%20cifar), along with movies showing how topography evolves during training of these networks, for the 3 categories considered.

## Discussion

### Classification performance
We have seen that Reset networks can classify standard computer vision datasets such as MNIST, Fashion MNIST, CIFAR-10 and CIFAR-100. However and as the figure below shows for the later 2 datasets, at this stage their performance is disappointing, only at best matching that of a single Resnet-20, while having many more parameters. 

![Reset performance Cifar10Cifar100](https://user-images.githubusercontent.com/13241166/143680476-ff8fd5eb-abce-40aa-8da9-a92edf0b0ed8.png)

One reason could be that in our simulations, spatial resets between levels were always done by reshaping the subnetworks' outputs, which constitute an information botteneck. Reshaping prior to the subnetwork's output, e.g. the dense layer or before, might be a more astute choice. We also observe that the full resources of the Reset network don't seem to be used: some subnetwork units are more active than others. This can be alleviated to some extent by using dropout, or another kind of regularization on the grid.

### Regularization by auto-encoding
In the course of our investigations, we observed that Reset networks that were based on smaller subnetworks than Reset20, performed much better when the second level had 2 networks: one that classified the input, and another that tried to reconstruct the input from the grid. As shown in the figure below, auto-encoding in this situation appears to act as an efficient regularizer for classification, forcing neural activity to be distributed across the whole grid rather than to be drawn by one, or just a few subnetworks.

![auto_encoding_regularization](https://user-images.githubusercontent.com/13241166/143776940-28693edd-ef46-42a7-9ccc-59727559592b.png)

Such regularization effects from autoencoding have been reported before for standard classifiers [4]. The novelty in Reset networks is that input reconstruction must be accomplished using the information from the whole grid: this suggests that in visual cortex, some feedback connections between distal cortical areas actually function as regularizers of cortical spaces.

### Topography
Reset networks constitute a novel mechanism for topography to emerge in deep learning. We have established that the networks self-organize for MNIST, Fashion MNIST and CIFAR. We then presented firm evidence that they could reproduce topographic organization in parietal cortex for realistic number codes, and in ventral Occipitotemporal cortex for the so-called "categorical areas". A related point is that Reset networks provide a way to implement a cortical gradient, the mapping between foveal/peripheral input and lateral/medial in visual cortex, which is not easily captured within the standard "one-CNN-fits-all" approach.

### Adding networks when necessary: the width and depth of Reset networks
Reset networks align well with a view of neural development in which, rather than or in addition to recycling extant neural material, neural resources can also be recruited in the system if needed. Learning a new task could thus require only to widen the system by adding a network at the current level, with different networks possibly trained on different tasks. If expertise from previously learned tasks is required, the system can be made deeper by reshaping network outputs at the current level and creating a new level.

## Conclusion
Reset networks show that topography can emerge in deep CNN classifiers, when composed with one another. In this view, the topographic cortex should not be modeled as a single classifier, however deep and richly organized, but as a sequence of levels of neural network classifiers. This rests on the idea that the cortex has the ability to compose networks with one another if need be, and predicts that the outputs, or the late computational stages, of cortical classifiers are either spatially organized, or somehow reshaped spatially during the course of composition.

## Citation
Hannagan T. Reset Networks: Emergent Topography by Composition of Convolutional Neural Networks. [bioRxiv 2021.11.19.469308](https://www.biorxiv.org/content/10.1101/2021.11.19.469308v3)


## References
[1] Patel GH, Kaplan DM, Snyder LH. Topographic organization in the brain: searching for general principles. Trends Cogn Sci. 2014;18(7):351-363. [doi:10.1016/j.tics.2014.03.008](https://pubmed.ncbi.nlm.nih.gov/24862252/)

[2] Hannagan T, Nieder A, Viswanathan P, Dehaene S. A random-matrix theory of the number sense. Phil. Trans. R. Soc. B. 2018;373:20170253.
[doi:10.1098/rstb.2017.0253](https://royalsocietypublishing.org/doi/10.1098/rstb.2017.0253)

[3] Lee H, Margalit E, Jozwik KM, Cohen MA, Kanwisher N, Yamins DL, DiCarlo JJ. Topographic deep artificial neural networks reproduce the 
hallmarks of the primate inferior temporal cortex face processing network. 2020 [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.07.09.185116v1.full.pdf).

[4] Le L, Patterson A, White M. Supervised autoencoders: Improving generalization performance with unsupervised regularizers. In [Advances in Neural Information Processing Systems. 2018. 107–117](https://proceedings.neurips.cc/paper/2018/file/2a38a4a9316c49e5a833517c45d31070-Paper.pdf).

[5] Hannagan T, Agrawal A., Cohen L, Dehaene S. Emergence of a compositional neural code for written words: Recycling of a convolutional neural network for reading. Proceedings of the National Academy of Sciences Nov 2021, 118 (46) e2104779118; [doi: 10.1073/pnas.2104779118](https://www.pnas.org/content/118/46/e2104779118)
