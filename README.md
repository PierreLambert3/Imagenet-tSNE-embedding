# Imagenet-tSNE-embedding

IMPORTANT: it seems a git clone doesn't pull the .npy files, but from the web interface of github, you can click on the file name and then click on the small "dowload" button to correctly download the files (156MB for X, 4.9MB for Y).
 

tSNE representation as mentioned in the paper "Neighbour Embeddings: Beyond Visualisation" by P. Lambert & colleagues, Machine Learning Methods in Visualisation for Big Data (2025) (https://diglib.eg.org/collections/d1dfb060-0146-49af-a115-9151bd985622)


The original representation of Imagenet was pulled from the latent space of the EVA vision-language model, that's a 1280-dimensional space. ("EVA: Exploring the Limits of Masked Visual Representation Learning at Scale")

The 1280 dimensions are reduced to 192 by PCA (explained variance = 71.1 percent)

The 192-dimensional dataset is then reduced to 32 by tSNE, using https://github.com/PierreLambert3/fast_htSNE

Here's the hyperparameters used by tSNE (increased attractive forces as is recomended on very large datasets, see "The art of using tSNE for single-cell transcriptomics" by Kobak & berens)

Xld, knn_HD = htSNE(n_components=32, verbose=True, with_gui=False).fit(X, Y, max_n_iter=1000,
                        purpose_is_KNN=False, base_attraction_repulsion_ratio=90.0, end_attrac_mult=0.9, end_kernel_alpha=1.0, lr_strength=16.0)

Likely improvements:
    - Use more than 192 PCs before tSNE  (but the implementation is limited to 256 dims max, for optimisation reasons related to GPUs)\n
    - Hyperparameter tuning on tSNE.\n
    - More iterations, here it is the result of just a couple of hundreds iterations.\n
    - If we could derive a (weakly supervised -) loss function for neural nets inspired by tSNE, that would be cool. For instance with diffusing labels.\n

