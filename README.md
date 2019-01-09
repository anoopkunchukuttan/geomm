# Geometry-aware Multilingual Embedding


Code for learning multilingual embeddings using the method reported in: 

Pratik Jawanpuria, Arjun Balgovind, Anoop Kunchukuttan, Bamdev Mishra. _[Learning Multilingual Word Embeddings in Latent Metric Space: A Geometric Approach](https://arxiv.org/abs/1808.08773)_. arxiv:1808.08773. 2018.

## Environment Setup
Do the following steps in order:
1. Clone the repository
2. Create a python virtual environment without Tensorflow (if TF is present Pymanopt gives out of memory errors).  
3. pip install numpy scipy ipdb
4. pip install git+https://github.com/pymanopt/pymanopt.git --upgrade
5. In Pymanopt code(located at C:\Anaconda\envs\ENVRNMT_NAME\Lib\site-packages\pymanopt\tools\autodiff for Windows or the Linux equivalent), at line 46,49,101,104 add a parameter to the call of theano.function, allow_input_downcast=True 
6. conda install theano pygpu
7. In Users\USER_NAME make a file .theanorc.txt with following content:

		[global]
		device = cuda
		floatX = float32
8. Install cupy based on your CUDA version

**Note:** *While using this setup with PyManopt, make sure to import cupy before importing theano, as sometimes theano throws an error that it is unable to find the correct CUDA version. However, the use of Cupy before this fixes the issue.*

## Datasets
The datasets can be downloaded by running the following commands in vecmap_data/ and muse_data/
		
	./get_vecmap_data.sh
	./get_muse_data.sh

## Reproducing Results

The results that have been reported in  _[Learning Multilingual Word Embeddings in Latent Metric Space: A Geometric Approach](https://arxiv.org/abs/1808.08773)_ can be reproduced by running the following scripts:
* Results of the GeoMM algorithm reported in Table 1, 2, and 6:
	
		./geomm_results.sh
* Results of the GeoMM-Multi algorithm reported in Table 1, 2, and 6:
	
		./geomm_multi_results.sh
* Results of the GeoMM-Semi algorithm reported in Table 7:
	
		./geomm_semi_results.sh

**Note:** Since our code makes use of CUDA and FP32 precision, it may not be possible to reproduce our results exactly, due to minor numerical variations in GPU operations. However, the effect on the final results is negligible, as we have observed the variations usually lie within an error margin of 0.1 or 0.2.

## GeoMM Embeddings 

**Coming soon**

We provide GeoMM bilingual and multilingual embeddings for monolingual embeddings in the FastText repository.

### Using CommonCrawl FastText Embeddings

#### English to Indian language bilingual embeddings

### Using Wikipedia FastText Embeddings



## Acknowledgements
The data-processing part of our code was taken from _[Artetxe's Vecmap Repository](https://github.com/artetxem/vecmap)_.

## References
Please cite _[Learning Multilingual Word Embeddings in Latent Metric Space: A Geometric Approach](https://arxiv.org/abs/1808.08773)_ if you found the resources in this repository useful.

	@article{jawanpuria2018learning,
	  title={Learning multilingual word embeddings in latent metric space: a geometric approach},
	  author={Jawanpuria, Pratik and Balgovind, Arjun and Kunchukuttan, Anoop and Mishra, Bamdev},
	  journal={Accepted in Transaction of the Association for Computational Linguistics (TACL)},
	  year={2019}
	}
