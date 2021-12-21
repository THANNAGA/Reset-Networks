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

Let's start by training 3 Reset(8) networks, each on a classic computer vision datasets: MNIST, Fashion MNIST and CIFAR-10. In all cases, the networks reached standard performance levels on the test sets, but more interestingly, the following figure shows the models' grids after 20 epochs of training.

![topography_across_domains](https://user-images.githubusercontent.com/13241166/146968368-446aa746-6a98-4e85-ad08-ddc2b37cc11f.png)

The upper plots present converged preference maps -the class preference of each unit on the 32x32 grid of the trained model- whereas the lower plots quantify the amount of clustering on each map, at each point during training. 

A unit's preference is given by the highest d-prime, over each target class, of the unit's responses to this target class against all other classes. If this maximum d-prime is below an arbitrary threshold, here set to 2, the unit is said to have no preference (white units on the preference map). Clustering is then defined as the average size, over all target classes, of the connected components present on the map for this class. More details can be found in the [paper](https://www.biorxiv.org/content/10.1101/2021.11.19.469308v3). The final clustering index presented in the lower plots is the deviation of clustering from chance: it is obtained by subtracting the clustering for shuffled maps to that of the true maps. There is clustering as soon as the index is above zero.

There is clustering for each of the three domains considered, with some variations across domains -for instance CIFAR-10 produces more clustering, while Fashion MNIST produces fewer category selective units. Clustering also happens quite quickly: it is largely in place after the first training epoch.

One might be forgiven to think that clustering only comes from the concatenation of reshaped outputs. To assess whether this is the case, we measure clustering in the family of Reset(n) networks for n = 1, 2, 4 and 8. The grid of Reset(1) has no concatenation, while that of Reset(8) is obtained by concatenating the output units of 64 subnetworks. We emphasize that although n varies, the size of the grid remains fixed at 32x32 units. 

![topography_across_domains_and_sizes](https://user-images.githubusercontent.com/13241166/146968544-182015f0-77b3-4463-835a-41439d7efc7c.png)

It can be seen that clustering is always non-zero for all n, and that there is a clear tendency of clustering to increase with n. Though a significant component of clustering comes from concatenating the outputs of several networks, it is not a necessary condition: the Reset(1) curve shows that just stacking two CNN classifiers on top of one another already will produce clustering.

## Reset networks and categorical areas in ventral occipitotemporal cortex

In ventral occipitotemporal cortex, more than two decades of studies have established the presence of areas selective for various widespread visual categories, in particular faces, bodies, tools, houses, and words [1]. While there is no shortage of computational models able to reproduce many caracteristics of the visual system, including some of the so-called Visual Word Form Area in vOTC [2], as of 2021 only one model [3] achieves both topography and scale at the same time - with topography being problematic as it requires two different notions of space to coexist. By contrast, the way Reset networks achieve topography at scale is conceptually straightforward. 
I thank ![Hyodong Lee](https://scholar.google.com/citations?user=1QvDhAQAAAAJ&hl=en), who would likely disagree with my assessment of the problem, for email exchanges on her innovative work on Topographic Deep Artificial Neural Networks. 

![topography_CIFAR100_across_sizes](https://user-images.githubusercontent.com/13241166/146969318-9564c726-d12a-483f-a166-db5bee3da980.png)

The figure above shows final preference maps of Reset(1), (2), (4) and (8) networks trained on Cifar-100, as well as their clustering indices throughout training. I tested the networks' preferences on 3 macro-categories - objects, houses and people - obtained by agregating the relevant cifar-100 classes. 

Though many units show no special preference for these macro-categories, clustering is still obvious on the maps, as it is in the mean cluster sizes. There are a few points worth noting. First, by and large clustering tends to increase with the number of epochs and with the size of the Reset network. Second, subnetworks tend to specialize for some categories. This is particularly well exemplified in the case of Reset(2), where the upper-left remains insensitive, the lower right subnetwork has units specializing for all 3 categories, while the lower left and upper right ones specialize for people and objects, respectively.

Pretrained Reset networks are available in the folder ![Topography for cifar](https://github.com/THANNAGA/Reset-Networks/tree/main/Topography%20for%20cifar), along with movies showing how topography evolves during training of these networks, for the 3 categories considered.

## Reset networks and topography for numbers in parietal cortex

So far we have been dealing with clustering, which, though related a notion, is weaker than topography: one can have clustering without topography, but topographic organization necessarily implies clustering. 

Cortical topography in the strict sense is the notion that "nearby neurons in the cortex have receptive fields at nearby locations in the world" [4]. However the term has come to take a wider meaning: it is often understood as applying also to local fields or voxels as well as to neurons, and to refer to any kind of selectivity, not just location selectivity. In this wider sense, topography is a widespread phenomenon in brain imaging, observed throughout the visual cortex as well as in some associative areas. 

In parietal cortex, voxels selective for similar numbers are more likely to be contiguous: this number topography is not yet well understood. Of course, we have just seen that Reset networks will self-organize when trained to classify hand-written digits, so one might argue that this is a good start for an explanation. However, classifying MNIST is not entirely satisfying from a neuroscientific point of view: it is very likely that kids map written digits not onto one-hot labels, but onto pre-existing (possibly innate!) number representations that have a rather special format. 

The nature of these codes has been thoroughly studied in [5]: a lot of phenomena could be explained if number codes were sparsely distributed vectors, with exponentially more overlap between successive number codes as numbers increase. Therefore, it would be more convincing if Reset networks could reproduce number topography by mapping digit images onto these realistic number codes (and even more so if the task was to map visual scenes with n objects to the number code for n, but that's yet another project).

As the figure below shows, a sequence of Reset(1), (2), (4) and (8) networks -all with the same grid size of 32x32 units- can be trained to map images of digits onto number codes, and succeeds in reproducing topographic organization. We emphasize that we would expect the same topography to emerge for classification of dots, or any kind of stimuli with countable objects.

![topography_Numbers10_across_sizes_3](https://user-images.githubusercontent.com/13241166/146969408-81fd3a7a-c002-441c-9c75-bcabfcb5dbcb.png)

Number topography is visible on the maps of number preferences in this figure (upper line), and is quantified in the clustering curves (middle line), where it can also be seen to emerge quickly during training. Cluster size (black curve) is always quite significantly above what it is for a shuffled (grey curve) selectivity map. Notably, there is a tendency of subnetworks to specialize for specific numbers, or numbers in the same ballpark. Consider the grid of Reset(2), whose lower left quadrant has a thing for numbers in the higher range (between 6 and 8), while its lower right quadrant specializes for small numbers 0 and 1. Also check out in this respect the videos for ![Reset(2)](https://github.com/THANNAGA/Reset-Networks/blob/main/Topography%20for%20numbers/history_monitor_ResetNumber2_Numbers10.mp4) and ![Reset(4)](https://github.com/THANNAGA/Reset-Networks/blob/main/Topography%20for%20numbers/history_monitor_ResetNumber4_Numbers10.mp4).

Because in this regression task, the codes onto which number images are mapped have non-degenerate similarities (unlike the degenerate similarities of one-hot labels in the previous tasks), one can expect not only clustering, but also some proper topography to emerge on the Reset networks' grids. It is then time to introduce a topographic index for our preference maps. We define topography as the average, over all units on the grid, of the proximity of a unit's number preference to those of its 8 immediate neighbors. The final topography index presented in the Figure's lower plots, then, is just the deviation of this topography from chance: it is obtained by subtracting the topography measured for shuffled maps to that of actual maps. There is topography as soon as the index is above zero.

The lower plots in this figure show that topography is widespread in Reset networks for this regression task. With the exception of Reset(1), topography is always much higher than the chance level of 0 (no topography). Two more effects also imediately stand out in the figure: topography tends to increase with training, as well as with n.

Again, more details can be found in the [paper](https://www.biorxiv.org/content/10.1101/2021.11.19.469308v3), and pretrained Reset networks are available in the folder ![Topography for numbers](https://github.com/THANNAGA/Reset-Networks/tree/main/Topography%20for%20numbers), where one can also find the above mentioned movies, showing how number topography evolves during training.

## Discussion

### Classification performance
We have seen that Reset networks can classify standard computer vision datasets such as MNIST, Fashion MNIST, CIFAR-10 and CIFAR-100. However and as the figure below shows for the later 2 datasets, at this stage their performance is disappointing, only at best matching that of a single Resnet-20, while having many more parameters. 

![Reset performance Cifar10Cifar100](https://user-images.githubusercontent.com/13241166/143680476-ff8fd5eb-abce-40aa-8da9-a92edf0b0ed8.png)

One reason could be that in our simulations, spatial resets between levels were always done by reshaping the subnetworks' outputs, which constitute an information botteneck. Reshaping prior to the subnetwork's output, e.g. the dense layer or before, might be a more astute choice. We also observe that the full resources of the Reset network don't seem to be used: some subnetwork units are more active than others. This can be alleviated to some extent by using dropout, or another kind of regularization on the grid.

### Regularization by auto-encoding
In the course of our investigations, we observed that Reset networks that were based on smaller subnetworks than Reset20, performed much better when the second level had 2 networks: one that classified the input, and another that tried to reconstruct the input from the grid. As shown in the figure below, auto-encoding in this situation appears to act as an efficient regularizer for classification, forcing neural activity to be distributed across the whole grid rather than to be drawn by one, or just a few subnetworks.

![auto_encoding_regularization](https://user-images.githubusercontent.com/13241166/143776940-28693edd-ef46-42a7-9ccc-59727559592b.png)

Such regularization effects from autoencoding have been reported before for standard classifiers [6]. The novelty in Reset networks is that input reconstruction must be accomplished using the information from the whole grid: this suggests that in visual cortex, some feedback connections between distal cortical areas actually function as regularizers of cortical spaces.

### Topography
Reset networks constitute a novel mechanism for topography to emerge in deep learning. We have established that the networks self-organize for MNIST, Fashion MNIST and CIFAR. We then presented firm evidence that they could reproduce topographic organization in parietal cortex for realistic number codes, and in ventral Occipitotemporal cortex for the so-called "categorical areas". A related point is that Reset networks provide a way to implement a cortical gradient, the mapping between foveal/peripheral input and lateral/medial in visual cortex, which is not easily captured within the standard "one-CNN-fits-all" approach.

### Adding networks when necessary: the width and depth of Reset networks
Reset networks align well with a view of neural development in which, rather than or in addition to recycling extant neural material, neural resources can also be recruited in the system if needed. Learning a new task could thus require only to widen the system by adding a network at the current level, with different networks possibly trained on different tasks. If expertise from previously learned tasks is required, the system can be made deeper by reshaping network outputs at the current level and creating a new level.

## Conclusion
Reset networks show that topography can emerge in deep CNN classifiers, when composed with one another. In this view, the topographic cortex should not be modeled as a single classifier, however deep and richly organized, but as a sequence of levels of neural network classifiers. This rests on the idea that the cortex has the ability to compose networks with one another if need be, and predicts that the outputs, or the late computational stages, of cortical classifiers are either spatially organized, or somehow reshaped spatially during the course of composition.

## Citation
Hannagan T. Reset Networks: Emergent Topography by Composition of Convolutional Neural Networks. [bioRxiv 2021.11.19.469308](https://www.biorxiv.org/content/10.1101/2021.11.19.469308v3)

## Note
Due to GitHub's limitations on upload size, I could not include pretrained Reset(8) networks in this repository.
On the other hand, there are folders for each of the tasks considered -MNIST, Fashion MNIST, Cifar-10, Cifar-100 and number codes- where one can find the notebooks used for training and analysis. The python code in these notebooks is not always optimized or clean, and there are some differences in the topography and clustering indices used in the anaylsis.

## References
[1] Grill-Spector K, Weiner KS. (2014) The functional architecture of the ventral temporal cortex and its role in categorization. Nat. Rev. Neurosci. 15, 536–548.[https://doi.org/10.1038/nrn3747](https://www.nature.com/articles/nrn3747.pdf?origin=ppub)

[2] Hannagan T, Agrawal A., Cohen L, Dehaene S. Emergence of a compositional neural code for written words: Recycling of a convolutional neural network for reading. Proceedings of the National Academy of Sciences Nov 2021, 118 (46) e2104779118; [doi: 10.1073/pnas.2104779118](https://www.pnas.org/content/118/46/e2104779118)

[3] Lee H, Margalit E, Jozwik KM, Cohen MA, Kanwisher N, Yamins DL, DiCarlo JJ. Topographic deep artificial neural networks reproduce the 
hallmarks of the primate inferior temporal cortex face processing network. 2020 [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.07.09.185116v1.full.pdf).

[4] Patel GH, Kaplan DM, Snyder LH. Topographic organization in the brain: searching for general principles. Trends Cogn Sci. 2014;18(7):351-363. [doi:10.1016/j.tics.2014.03.008](https://pubmed.ncbi.nlm.nih.gov/24862252/)

[5] Hannagan T, Nieder A, Viswanathan P, Dehaene S. A random-matrix theory of the number sense. Phil. Trans. R. Soc. B. 2018;373:20170253.
[doi:10.1098/rstb.2017.0253](https://royalsocietypublishing.org/doi/10.1098/rstb.2017.0253)

[6] Le L, Patterson A, White M. Supervised autoencoders: Improving generalization performance with unsupervised regularizers. In [Advances in Neural Information Processing Systems. 2018. 107–117](https://proceedings.neurips.cc/paper/2018/file/2a38a4a9316c49e5a833517c45d31070-Paper.pdf).


