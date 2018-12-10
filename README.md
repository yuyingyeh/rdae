# Interactive Reconstruction of Monte Carlo Image Sequences using a Recurrent Denoising Autoencoder
Implementation and some visual results of SIGGRAPH'17 paper: [Interactive Reconstruction of Monte Carlo Image Sequences using a Recurrent Denoising Autoencoder](https://research.nvidia.com/sites/default/files/publications/dnn_denoise_author.pdf) by Chakravarty R. Alla Chaitanya, Anton Kaplanyan, Christoph Schied, Marco Salvi, Aaron Lefohn, Derek Nowrouzezahrai and Timo Aila

## Architecture
We use the same architecture proposed in the paper instead we change the recurrent block to [ConvLSTM](https://arxiv.org/pdf/1506.04214.pdf) blocks.

## Data Acquisition
We use [pbrt-v3](https://www.pbrt.org/fileformat-v3.html) to render 1 spp, 4096 spp, and auxiliary features as training pairs.

![training data]()

## Visual Results
![video](https://github.com/.gif "video")
