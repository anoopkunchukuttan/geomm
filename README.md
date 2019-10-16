# Geometry-aware Multilingual Embedding


Code for learning multilingual embeddings using the method reported in: 

Pratik Jawanpuria, Arjun Balgovind, Anoop Kunchukuttan, Bamdev Mishra. _[Learning Multilingual Word Embeddings in Latent Metric Space: A Geometric Approach](https://www.mitpressjournals.org/doi/full/10.1162/tacl_a_00257)_. Transaction of the Association for Computational Linguistics (TACL), Volume 7, p.107-120, 2019.

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
9. Two GPUs are needed

**Note:** *While using this setup with Pymanopt, make sure to import cupy before importing theano, as sometimes theano throws an error that it is unable to find the correct CUDA version. However, the use of Cupy before this fixes the issue.*

## Datasets
The datasets can be downloaded by running the following commands in vecmap_data/ and muse_data/
		
	./get_vecmap_data.sh
	./get_muse_data.sh

## Reproducing Results

The results that have been reported in  _[Learning Multilingual Word Embeddings in Latent Metric Space: A Geometric Approach](https://www.mitpressjournals.org/doi/full/10.1162/tacl_a_00257)_ can be reproduced by running the following scripts:
* Results of the GeoMM algorithm reported in Table 1, 2, and 6:
	
		./geomm_results.sh
* Results of the GeoMM-Multi algorithm reported in Table 1, 2, and 6:
	
		./geomm_multi_results.sh
* Results of the GeoMM-Semi algorithm reported in Table 7:
	
		./geomm_semi_results.sh

**Note:** Since our code makes use of CUDA and FP32 precision, it may not be possible to reproduce our results exactly, due to minor numerical variations in GPU operations. However, the effect on the final results is negligible, as we have observed the variations usually lie within an error margin of 0.1 or 0.2.

**Note:** Added geomm_optimized.py which can replace geomm.py in all use-cases. Reduces time-taken for en-es pair from 188.5 second to 6.5 second.

## GeoMM Embeddings 

We provide GeoMM bilingual and multilingual embeddings. These are normalized embeddings in the latent space, <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\phi(w)=||\mathbf{B}^{\frac{1}{2}}w||_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\phi(w)=||\mathbf{B}^{\frac{1}{2}}w||_2" title="\phi(w)=||\mathbf{B}^{\frac{1}{2}}\mathbf{U}w||_2" /></a>. The embeddings are made available under the following license: <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>. 

### MUSE Dataset

These embeddings have been trained jointly using en-XX [MUSE bilingual dictionaries](https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries) and [Wikipedia FastText embeddings](https://fasttext.cc/docs/en/pretrained-vectors.html).  

|||||||
|---|---|---|---|---|---|
| [de](https://akpublicdata.blob.core.windows.net/publicdata/geomm/muse/muse-de.vec.gz) | [en](https://akpublicdata.blob.core.windows.net/publicdata/geomm/muse/muse-en.vec.gz) | [es](https://akpublicdata.blob.core.windows.net/publicdata/geomm/muse/muse-es.vec.gz) | [fr](https://akpublicdata.blob.core.windows.net/publicdata/geomm/muse/muse-fr.vec.gz) | [ru](https://akpublicdata.blob.core.windows.net/publicdata/geomm/muse/muse-ru.vec.gz) | [zh](https://akpublicdata.blob.core.windows.net/publicdata/geomm/muse/muse-zh.vec.gz) |

### VecMap Dataset

These embeddings have been trained jointly using en-XX bilingual dictionaries and embeddings from the [VecMap dataset](https://github.com/artetxem/vecmap/blob/master/get_data.sh). 

||||||
|---|---|---|---|---|
| [de](https://akpublicdata.blob.core.windows.net/publicdata/geomm/vecmap/vecmap-de.vec.gz) | [en](https://akpublicdata.blob.core.windows.net/publicdata/geomm/vecmap/vecmap-en.vec.gz) | [es](https://akpublicdata.blob.core.windows.net/publicdata/geomm/vecmap/vecmap-es.vec.gz) | [fi](https://akpublicdata.blob.core.windows.net/publicdata/geomm/vecmap/vecmap-fi.vec.gz) | [it](https://akpublicdata.blob.core.windows.net/publicdata/geomm/vecmap/vecmap-it.vec.gz) |

#### English-Indian language bilingual embeddings

These bilingual embeddings have been trained using the [CommonCrawl+Wikipedia FastText Embeddings](https://fasttext.cc/docs/en/crawl-vectors.html) and the [MUSE bilingual dictionaries](https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries).

||||
|---|---|---|
| [en-hi](https://akpublicdata.blob.core.windows.net/publicdata/geomm/cc/en-hi.tgz) | [en-bn](https://akpublicdata.blob.core.windows.net/publicdata/geomm/cc/en-bn.tgz) | [en-ta](https://akpublicdata.blob.core.windows.net/publicdata/geomm/cc/en-ta.tgz) |  


## Acknowledgements
The data-processing part of our code was taken from _[Mikel Artetxe's Vecmap Repository](https://github.com/artetxem/vecmap)_.

## References
Please cite _[Learning Multilingual Word Embeddings in Latent Metric Space: A Geometric Approach](https://www.mitpressjournals.org/doi/full/10.1162/tacl_a_00257)_ if you found the resources in this repository useful.

	@article{jawanpuria2018learning,
	  title={Learning multilingual word embeddings in latent metric space: a geometric approach},
	  author={Jawanpuria, Pratik and Balgovind, Arjun and Kunchukuttan, Anoop and Mishra, Bamdev},
	  journal={Transaction of the Association for Computational Linguistics (TACL)},
	  volume={7},
	  pages={107--120},
	  year={2019}
	}
