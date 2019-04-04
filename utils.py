import numpy as np
import cupy as cp
import embeddings
import scipy.linalg
import collections 
import sys
import time
import os
from subprocess import Popen, PIPE, STDOUT
import shlex
import numpy as np


def read_model(mapping_model_dir):
    """
        Reads the model and returns a dictionary with model parameters

    """    

    model_params={}

    model_params['U_src']=np.loadtxt(mapping_model_dir+'/U_src.csv')
    model_params['U_tgt']=np.loadtxt(mapping_model_dir+'/U_tgt.csv')
    model_params['B']=np.loadtxt(mapping_model_dir+'/B.csv')

    return model_params

def apply_mapping(x,vocab_type,model_params, latent_space=True): 
    """
     Applies bilingual mapping to the matrix x and returns the transformed matrix. 
     
     vocab_type is one of `src` or `tgt`

     latent_space: If true, the embeddings are mapped to latent space. Otherwise, 
        they are mapped to the embedding space of the other language. 
    """
   
    xw=None 

    if vocab_type=='src':
        if latent_space:
            xw = x.dot( model_params['U_src'] ).dot(scipy.linalg.sqrtm( model_params['B']  ))
        else:
            xw = x.dot( model_params['U_src'] ).dot( model_params['B'] ).dot( model_params['U_tgt'].T )
    elif vocab_type=='tgt':
        if latent_space:
            xw = x.dot( model_params['U_tgt'] ).dot(scipy.linalg.sqrtm( model_params['B']  ))
        else:
            xw = x.dot( model_params['U_tgt'] ).dot( model_params['B'] ).dot( model_params['U_src'].T )

    return xw    

def map_embedding_db(in_emb_fname, out_emb_fname, vocab_type, mapping_model_dir, latent_space=True): 
    """
    Maps all the vocabulary in `in_emb_fname` to target language space using the model in `mapping_model_dir`
    """

    print('Loading train data...')
    # Read input embeddings
    with open(in_emb_fname, 'r', encoding='utf-8', errors='surrogateescape') as srcfile:
        src_words, x = embeddings.read(srcfile,max_voc=0, dtype='float32')
        src_word2ind = {word: i for i, word in enumerate(src_words)}

    model_params=read_model(mapping_model_dir)
    xw=apply_mapping(x,vocab_type,model_params,latent_space)

    with open(out_emb_fname,'w',encoding='utf-8') as outfile:
        embeddings.write(src_words,xw,outfile)

def translate(src_emb_fname, tgt_emb_fname, trans_tgt_fname, trans_src_fname=None, retrieval_method="csls", csls_k=10, batch_size=2500):

    print('Loading train data...')

    srcfile=open(src_emb_fname, 'r', encoding='utf-8', errors='surrogateescape')
    tgtfile=open(tgt_emb_fname, 'r', encoding='utf-8', errors='surrogateescape')

    # Read source embeddings
    src_words, x = embeddings.read(srcfile,max_voc=0, dtype='float32')
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    
    # Read target embeddings
    tgt_words, z = embeddings.read(tgtfile,max_voc=0, dtype='float32')
    tgt_word2ind = {word: i for i, word in enumerate(tgt_words)}
   
    srcfile.close()
    tgtfile.close()

    xw = embeddings.length_normalize(x)
    zw = embeddings.length_normalize(z)

    all_words=[]
    trans_words=[]
    trans_idx=[]
    oov=set()
    if trans_src_fname is not None:
        with open(trans_src_fname, 'r', encoding='utf-8', errors='surrogateescape') as trans_src_file:
            for line in trans_src_file:
                try:
                    #w=line.strip().lower()
                    w=line.strip()
                    all_words.append(w)
                    w_ind = src_word2ind[w]
                    trans_words.append(w)
                    trans_idx.append(w_ind)
                except KeyError:
                    oov.add(w)
    else:                            
        all_words=src_words
        trans_words=src_words
        trans_idx=list(range(len(src_words)))
        oov=set()
  
    print(len(all_words))
    print(len(trans_words))
    print(len(trans_idx))
    print(len(oov))
    src=trans_idx 

    translation = collections.defaultdict(int)
    translation5 = collections.defaultdict(list)
    translation10 = collections.defaultdict(list)

    if retrieval_method == 'nn':  # Standard nearest neighbor
        for i in range(0, len(src), batch_size):
            j = min(i + batch_size, len(src))
            similarities = xw[src[i:j]].dot(zw.T)
            nn = similarities.argmax(axis=1).tolist()
            similarities_idx = similarities.argsort(axis=1)
            nn5 = similarities_idx[:,-5:]
            nn10 = similarities_idx[:,-10:]

            for k in range(j-i):
                translation[src[i+k]] = nn[k]
                translation5[src[i+k]] = nn5[k]
                translation10[src[i+k]] = nn10[k]
                
    elif retrieval_method == 'csls':
        t=time.time()
        nbrhood_x=np.zeros(xw.shape[0])
        nbrhood_z=np.zeros(zw.shape[0])
        nbrhood_z2=cp.zeros(zw.shape[0])
        print('Computing X Neighbourhood')
        # batch_size=1000
        for i in range(0, len(src), batch_size):
            j = min(i + batch_size, len(src))
            similarities = xw[src[i:j]].dot(zw.T)
            # similarities_x = np.sort(similarities, axis=1)
            similarities_x = -1*np.partition(-1*similarities,csls_k-1 ,axis=1)
            #similarities_x = -1*cp.partition(-1*cp.dot(cp.asarray(xw[src[i:j]]),cp.transpose(cp.asarray(zw))),csls_k-1 ,axis=1)[:,:csls_k]
            nbrhood_x[src[i:j]]=np.mean(similarities_x[:,:csls_k],axis=1)
        print('Completed in {0} seconds'.format(time.time()-t))
        print('Computing Z Neighbourhood')

        batch_num=1
        for i in range(0, zw.shape[0], batch_size):
            j = min(i + batch_size, zw.shape[0])
            similarities = -1*cp.partition(-1*cp.dot(cp.asarray(zw[i:j]),cp.transpose(cp.asarray(xw))),csls_k-1 ,axis=1)[:,:csls_k]
            nbrhood_z2[i:j]=(cp.mean(similarities[:,:csls_k],axis=1))
            print('Completed batch {0} in {1}'.format(batch_num, time.time()-t))
            batch_num+=1
        # gc.collect()
        # t=time.time()
        nbrhood_z=cp.asnumpy(nbrhood_z2)
        # ipdb.set_trace()
        print(time.time()-t)
        csls_alpha=1
        for i in range(0, len(src), batch_size):
            j = min(i + batch_size, len(src))
            similarities = xw[src[i:j]].dot(zw.T)
            similarities = np.transpose(np.transpose(2*similarities) - csls_alpha*nbrhood_x[src[i:j]])- csls_alpha*nbrhood_z
            nn = similarities.argmax(axis=1).tolist()
            print(time.time()-t)
            similarities = np.argsort((similarities),axis=1)

            nn5 = (similarities[:,-5:])
            nn10 = (similarities[:,-10:])
            for k in range(j-i):
                translation[src[i+k]] = nn[k]
                translation5[src[i+k]] = nn5[k]
                translation10[src[i+k]] = nn10[k]
        print('Completed in {0} seconds'.format(time.time()-t))

   
    ### write the translations (1 pair per line format)
    with open(trans_tgt_fname, 'w', encoding='utf-8', errors='surrogateescape') as trans_tgt_file:
        for w in trans_words: 
            trans=''
            if w in src_word2ind:
                trans=tgt_words[translation[src_word2ind[w]]]
            trans_tgt_file.write('{}\t{}\n'.format(w,trans))
   
    ### output in moses format 
    #with open(trans_tgt_fname, 'w', encoding='utf-8') as trans_tgt_file:
    #    for wno, w in enumerate(trans_words): 
    #        if w in src_word2ind:
    #            for t_idx in reversed(translation10[src_word2ind[w]]):
    #                trans_tgt_file.write( u'{} ||| {} ||| {} ||| {}\n'.format( 
    #                    wno, tgt_words[t_idx], '0.0 0.0 0.0 0.0', 0.0) )
    #        else: 
    #            trans_tgt_file.write( u'{} ||| {} ||| {} ||| {}\n'.format( 
    #                            wno, '@@OOV@@', '0.0 0.0 0.0 0.0', 0.0) )

def read_dict(fname,delim=' '):
    trans_dict={}
    with open(fname,'r',encoding='utf-8') as infile:
        for i,line in enumerate(infile,1):
            r=line.strip().split(delim)
            if len(r)!=2:
                print('Ignore entry on line {}: does not have 2 fields'.format(i))
                continue
            k,v=r
            trans_dict[k]=v

    return trans_dict

def compute_fasttext_embeddings(oov_words, model_path, fast_text_binary_path, dtype='float'):
    """
    Computes fasttext embeddings for given words. 
    Uses the `fasttext print-word-vectors` CLI interface for generating embeddings.
    
    oov_words: list of words to compute fasttext embeddings
    model_path: path to fasttext model (.bin file)
    fast_text_binary_path: path to fasttext binary
    """
    p = Popen(shlex.split('{} print-word-vectors {}'.format(fast_text_binary_path, model_path)), 
              stdout=PIPE, stdin=PIPE, stderr=PIPE,universal_newlines=True)
    
    stdout_data, stderr = p.communicate(input='\n'.join(oov_words))
    
    if len(stderr) > 0:
        raise Exception('Error running fasttext')
        
    emb_rows=[]
    words=[]
        
    for line in stdout_data.split('\n'):
        if len(line)>0:
            word, vec = line.strip().split(' ', 1)
            words.append(word)
            emb_rows.append(np.fromstring(vec, sep=' ', dtype=dtype))
        
    return (words, np.array(emb_rows, dtype=dtype))

def add_oov_embeddings(train_dict_fname, test_dict_fname, 
                       src_emb_fname, tgt_emb_fname,
                       out_src_emb_fname, out_tgt_emb_fname,
                       src_model_path, tgt_model_path,
                       fast_text_binary_path,
                       max_voc=200000,
                       emb_format='txt'):
    """
    Adds the embeddings for OOV words in the training and test dictionaries to the embedding file. 
    This is done by computing the embeddings using FastText. So, this method applies to FastText 
    embeddings only. Note that the output embedding file will contain only the OOV words plus 
    the first max_voc words in the original embedding file.
    
    train_dict_fname: 
    test_dict_fname: 
    src_emb_fname: embedding file for source language 
    tgt_emb_fname: embedding file for target language
    out_src_emb_fname: output embedding file for source language 
    out_tgt_emb_fname: output embedding file for target language    
    src_model_path: fasttext model for source language 
    tgt_model_path: fasttext model for targetqa language 
    fast_text_binary_path: path to fasttext binary
    max_voc: number of vocab items to process from the embedding file
    emb_format: format of embedding files. Currently supported: 'txt' - standard fast text format
    """
    
    ## read dictionaries
    train_dict = read_dict(train_dict_fname)    
    test_dict  = read_dict(test_dict_fname)
    
    # read embeddings 
    src_vcb_words=None
    src_emb=None
    tgt_vcb_words=None
    tgt_emb=None
    
    with open(src_emb_fname, 'r', encoding='utf-8' ) as src_emb_file, \
         open(tgt_emb_fname, 'r', encoding='utf-8' ) as tgt_emb_file:        
        src_vcb_words, src_emb = embeddings.read(src_emb_file, max_voc)
        tgt_vcb_words, tgt_emb = embeddings.read(tgt_emb_file, max_voc)
    
    ## find OOVs
    src_oov_words=set()
    src_oov_words.update(train_dict.keys())
    src_oov_words.update(test_dict.keys())
    src_oov_words.difference_update(src_vcb_words)
    print('Number of src OOV words: {}'.format(len(src_oov_words)))
    
    tgt_oov_words=set()
    tgt_oov_words.update(train_dict.values())
    tgt_oov_words.update(test_dict.values())   
    tgt_oov_words.difference_update(tgt_vcb_words) 
    print('Number of tgt OOV words: {}'.format(len(tgt_oov_words)))
    
    ## compute embeddings for OOV
    ##### cat queries.txt | ./fasttext print-word-vectors model.bin
    src_oov_final_words, src_oov_emb = compute_fasttext_embeddings(src_oov_words, src_model_path, fast_text_binary_path)
    tgt_oov_final_words, tgt_oov_emb = compute_fasttext_embeddings(tgt_oov_words, tgt_model_path, fast_text_binary_path)
    
    if(len(src_oov_words)!=len(src_oov_final_words)):
        print('WARNING: Embeddings not computed for {} words out of {} OOV source words'.format(
            len(src_oov_words)-len(src_oov_final_words),
            len(src_oov_words)))
        
    if(len(tgt_oov_words)!=len(tgt_oov_final_words)):
        print('WARNING: Embeddings not computed for {} words out of {} OOV target words'.format(
            len(tgt_oov_words)-len(tgt_oov_final_words),
            len(tgt_oov_words)))        
    
    ## write new embeddings files to disk
    ## put the OOV words first followed by words in the original embeddings file 
    with open(out_src_emb_fname, 'w', encoding='utf-8' ) as out_src_emb_file, \
         open(out_tgt_emb_fname, 'w', encoding='utf-8' ) as out_tgt_emb_file:       
        embeddings.write( src_oov_final_words+src_vcb_words, np.concatenate([src_oov_emb, src_emb]), out_src_emb_file )
        embeddings.write( tgt_oov_final_words+tgt_vcb_words, np.concatenate([tgt_oov_emb, tgt_emb]), out_tgt_emb_file )   
    
