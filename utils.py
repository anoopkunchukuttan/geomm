import numpy as np
import cupy as cp
import embeddings
import scipy.linalg
import collections 
import sys
import time
import os

def read_model(mapping_model_dir):
    """
        Reads the model and returns a dictionary with model parameters

    """    

    model_params={}

    model_params['U_src']=np.loadtxt(mapping_model_dir+'/U_src.csv')
    model_params['U_tgt']=np.loadtxt(mapping_model_dir+'/U_tgt.csv')
    model_params['B']=np.loadtxt(mapping_model_dir+'/B.csv')

    return model_params

def apply_mapping(x,vocab_type,model_params): 
    """
     Applies bilingual mapping to the matrix x and returns the transformed matrix. 
     
     vocab_type is one of `src` or `tgt`
    """
   
    xw=None 

    if vocab_type=='src':
        xw = x.dot( model_params['U_src'] ).dot(scipy.linalg.sqrtm( model_params['B']  ))
    elif vocab_type=='tgt':
        xw = x.dot( model_params['U_tgt'] ).dot(scipy.linalg.sqrtm( model_params['B']  ))

    return xw    

def map_embedding_db(in_emb_fname, out_emb_fname, vocab_type, mapping_model_dir): 
    """
    Maps all the vocabulary in `in_emb_fname` to target language space under the `mapping_method` 
    using the model in `mapping_model_dir`
    """

    print('Loading train data...')
    # Read input embeddings
    with open(in_emb_fname, 'r', encoding='utf-8', errors='surrogateescape') as srcfile:
        src_words, x = embeddings.read(srcfile,max_voc=0, dtype='float32')
        src_word2ind = {word: i for i, word in enumerate(src_words)}

    model_params=read_model(mapping_model_dir)
    xw=apply_mapping(x,vocab_type,model_params)

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


