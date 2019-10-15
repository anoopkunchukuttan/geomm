import sys
import numpy as np
import cupy as cp
import embeddings
import scipy.linalg
import collections 
import time
import os
from subprocess import Popen, PIPE, STDOUT
import shlex
import numpy as np
import scipy.stats
import pygtrie
import operator
import itertools as it

# def read_model(mapping_model_dir):
#     """
#         Reads the model and returns a dictionary with model parameters

#     """    

#     model_params={}

#     model_params['U_src']=np.loadtxt(mapping_model_dir+'/U_src.csv')
#     model_params['U_tgt']=np.loadtxt(mapping_model_dir+'/U_tgt.csv')
#     model_params['B']=np.loadtxt(mapping_model_dir+'/B.csv')

#     return model_params

def read_model(mapping_model_dir):
    """
        Reads the model and returns a dictionary with model parameters

    """    

    model_params={}

    for f in os.listdir(mapping_model_dir):
        if f.find('U')==0 or f.find('B')==0:
            model_params[f.replace('.csv','')]=np.loadtxt( '{}/{}'.format(mapping_model_dir,f) )

    return model_params

# def apply_mapping(x,vocab_type,model_params, latent_space=True): 
#     """
#      Applies bilingual mapping to the matrix x and returns the transformed matrix. 
     
#      vocab_type is one of `src` or `tgt`. Indicates the source or target language as per the trained model. 

#      latent_space: If true, the embeddings are mapped to latent space. Otherwise, 
#         they are mapped to the embedding space of the other language. 
#     """
   
#     xw=None 

#     if vocab_type=='src':
#         if latent_space:
#             xw = x.dot( model_params['U_src'] ).dot(scipy.linalg.sqrtm( model_params['B']  ))
#         else:
#             xw = x.dot( model_params['U_src'] ).dot( model_params['B'] ).dot( model_params['U_tgt'].T )
#     elif vocab_type=='tgt':
#         if latent_space:
#             xw = x.dot( model_params['U_tgt'] ).dot(scipy.linalg.sqrtm( model_params['B']  ))
#         else:
#             xw = x.dot( model_params['U_tgt'] ).dot( model_params['B'] ).dot( model_params['U_src'].T )

#     return xw    

def apply_mapping(x, model_params, src='src', tgt='tgt', latent_space=True): 
    """
     Applies bilingual mapping to the matrix x and returns the transformed matrix. 
     
     vocab_type is one of `src` or `tgt`. Indicates the source or target language as per the trained model. 

     latent_space: If true, the embeddings are mapped to latent space. Otherwise, 
        they are mapped to the embedding space of the other language. 
    """
   
    xw=None 

    src_mat_name='U_{}'.format(src)
    tgt_mat_name='U_{}'.format(tgt)
    
    if latent_space:
        xw = x.dot( model_params[src_mat_name] ).dot(scipy.linalg.sqrtm( model_params['B']  ))
    else:
        xw = x.dot( model_params[src_mat_name] ).dot( model_params['B'] ).dot( model_params[tgt_mat_name].T )   
    
    return xw    

def build_w2i(words):
    return {word: i for i, word in enumerate(words)}

def translate(words_to_translate, src_emb_info, tgt_emb_info, retrieval_method="csls", csls_k=10, batch_size=2500):

    print('Hello')
    sys.stdout.flush()
    # Read source embeddings
    src_words, x = src_emb_info
    src_word2ind = build_w2i(src_words)
    
    # Read target embeddings
    tgt_words, z = tgt_emb_info
    tgt_word2ind = build_w2i(tgt_words)
   
    xw = embeddings.length_normalize(x)
    zw = embeddings.length_normalize(z)

    all_words=[]
    trans_words=[]
    trans_idx=[]
    oov=set()
    
    for w in words_to_translate:
        try:
            all_words.append(w)
            w_ind = src_word2ind[w]
            trans_words.append(w)
            trans_idx.append(w_ind)
        except KeyError:
            oov.add(w)
  
    print(len(all_words))
    print(len(trans_words))
    print(len(trans_idx))
    print(len(oov))
    src=trans_idx 
    
    print('Number of words to translate: {}'.format(len(src)))

    translation = collections.defaultdict(int)
    translation5 = collections.defaultdict(list)
    translation10 = collections.defaultdict(list)

    if retrieval_method == 'nn':  # Standard nearest neighbor
        for i in range(0, len(src), batch_size):
            j = min(i + batch_size, len(src))
            similarities = xw[src[i:j]].dot(zw.T)
            nn = similarities.argmax(axis=1).tolist()
#             similarities_idx = similarities.argsort(axis=1)
#             nn5 = similarities_idx[:,-5:]
#             nn10 = similarities_idx[:,-10:]

            for k in range(j-i):
                translation[src[i+k]] = nn[k]
#                 translation5[src[i+k]] = nn5[k]
#                 translation10[src[i+k]] = nn10[k]
                
    elif retrieval_method == 'csls':
        t=time.time()
        nbrhood_x=np.zeros(xw.shape[0])
        nbrhood_z=np.zeros(zw.shape[0])
        nbrhood_z2=cp.zeros(zw.shape[0])
        print('Computing X Neighbourhood')
        sys.stdout.flush()
        # batch_size=1000
        batch_num=1
        for i in range(0, len(src), batch_size):
            j = min(i + batch_size, len(src))
            similarities = xw[src[i:j]].dot(zw.T)
            # similarities_x = np.sort(similarities, axis=1)
            similarities_x = -1*np.partition(-1*similarities,csls_k-1 ,axis=1)
            #similarities_x = -1*cp.partition(-1*cp.dot(cp.asarray(xw[src[i:j]]),cp.transpose(cp.asarray(zw))),csls_k-1 ,axis=1)[:,:csls_k]
            nbrhood_x[src[i:j]]=np.mean(similarities_x[:,:csls_k],axis=1)
            print('Completed batch {0} in {1}'.format(batch_num, time.time()-t))
            sys.stdout.flush()
            batch_num+=1            
        print('Completed in {0} seconds'.format(time.time()-t))
        
        
        print('Computing Z Neighbourhood')
        sys.stdout.flush()

        batch_num=1
        for i in range(0, zw.shape[0], batch_size):
            j = min(i + batch_size, zw.shape[0])
            similarities = -1*cp.partition(-1*cp.dot(cp.asarray(zw[i:j]),cp.transpose(cp.asarray(xw))),csls_k-1 ,axis=1)[:,:csls_k]
            nbrhood_z2[i:j]=(cp.mean(similarities[:,:csls_k],axis=1))
            print('Completed batch {0} in {1}'.format(batch_num, time.time()-t))
            sys.stdout.flush()
            batch_num+=1
        # gc.collect()
        # t=time.time()
        nbrhood_z=cp.asnumpy(nbrhood_z2)
        # ipdb.set_trace()
        print(time.time()-t)
        print('Computing nearest neighbours')
        sys.stdout.flush()
        csls_alpha=1
        batch_num=1
        for i in range(0, len(src), batch_size):
            j = min(i + batch_size, len(src))
            similarities = xw[src[i:j]].dot(zw.T)
            similarities = np.transpose(np.transpose(2*similarities) \
                                        - csls_alpha*nbrhood_x[src[i:j]]) \
                                        - csls_alpha*nbrhood_z 
                
            nn = similarities.argmax(axis=1).tolist()
#             similarities = np.argsort((similarities),axis=1)
#             nn5 = (similarities[:,-5:])
#             nn10 = (similarities[:,-10:])
            for k in range(j-i):
                translation[src[i+k]] = nn[k]
#                 translation5[src[i+k]] = nn5[k]
#                 translation10[src[i+k]] = nn10[k]
                
            print('Completed batch {0} in {1}'.format(batch_num, time.time()-t))
            sys.stdout.flush()
            batch_num+=1
        print('Completed in {0} seconds'.format(time.time()-t))
        sys.stdout.flush()

    # get translations
    trans_pairs=[]
    for w in trans_words: 
        trans=''
        if w in src_word2ind:
            trans=tgt_words[translation[src_word2ind[w]]]
            trans_pairs.append((w,trans))

    return dict(trans_pairs)
   
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


# def translate(words_to_translate, src_emb_info, tgt_emb_info, retrieval_method="csls", csls_k=10, batch_size=2500):

#     print('Hello')
#     sys.stdout.flush()
#     # Read source embeddings
#     src_words, x = src_emb_info
#     src_word2ind = build_w2i(src_words)
    
#     # Read target embeddings
#     tgt_words, z = tgt_emb_info
#     tgt_word2ind = build_w2i(tgt_words)
   
#     xw = embeddings.length_normalize(x)
#     zw = embeddings.length_normalize(z)

#     all_words=[]
#     trans_words=[]
#     trans_idx=[]
#     oov=set()
    
#     for w in words_to_translate:
#         try:
#             all_words.append(w)
#             w_ind = src_word2ind[w]
#             trans_words.append(w)
#             trans_idx.append(w_ind)
#         except KeyError:
#             oov.add(w)
  
#     print(len(all_words))
#     print(len(trans_words))
#     print(len(trans_idx))
#     print(len(oov))
#     src=trans_idx 

#     translation = collections.defaultdict(int)
#     translation5 = collections.defaultdict(list)
#     translation10 = collections.defaultdict(list)

#     if retrieval_method == 'nn':  # Standard nearest neighbor
#         for i in range(0, len(src), batch_size):
#             j = min(i + batch_size, len(src))
#             similarities = xw[src[i:j]].dot(zw.T)
#             nn = similarities.argmax(axis=1).tolist()
#             similarities_idx = similarities.argsort(axis=1)
#             nn5 = similarities_idx[:,-5:]
#             nn10 = similarities_idx[:,-10:]

#             for k in range(j-i):
#                 translation[src[i+k]] = nn[k]
#                 translation5[src[i+k]] = nn5[k]
#                 translation10[src[i+k]] = nn10[k]
                
#     elif retrieval_method == 'csls':
#         t=time.time()
#         nbrhood_x=np.zeros(xw.shape[0])
#         nbrhood_z=np.zeros(zw.shape[0])
#         nbrhood_z2=cp.zeros(zw.shape[0])
        
#         print('Computing Z Neighbourhood')
#         sys.stdout.flush()

#         batch_num=1
#         for i in range(0, zw.shape[0], batch_size):
#             j = min(i + batch_size, zw.shape[0])
#             similarities = -1*cp.partition(-1*cp.dot(cp.asarray(zw[i:j]),cp.transpose(cp.asarray(xw))),csls_k-1 ,axis=1)[:,:csls_k]
#             nbrhood_z2[i:j]=(cp.mean(similarities[:,:csls_k],axis=1))
#             print('Completed batch {0} in {1}'.format(batch_num, time.time()-t))
#             sys.stdout.flush()
#             batch_num+=1
#         # gc.collect()
#         # t=time.time()
#         nbrhood_z=cp.asnumpy(nbrhood_z2)
        
        
#         print('Starting batchwise processing of souce words for finding best translation')
#         sys.stdout.flush()
        
#         csls_alpha=1
#         batch_num=1
#         for i in range(0, len(src), batch_size):
            
#             print('Start batch {0} in {1}'.format(batch_num, time.time()-t))
#             j = min(i + batch_size, len(src))
            
#             print('Computing X Neighbourhood')

#             similarities = xw[src[i:j]].dot(zw.T)
#             # similarities_x = np.sort(similarities, axis=1)
#             similarities_x = -1*np.partition(-1*similarities,csls_k-1 ,axis=1)
#             #similarities_x = -1*cp.partition(-1*cp.dot(cp.asarray(xw[src[i:j]]),cp.transpose(cp.asarray(zw))),csls_k-1 ,axis=1)[:,:csls_k]
#             nbrhood_x[src[i:j]]=np.mean(similarities_x[:,:csls_k],axis=1)
            
# #             print('Completed in {0} seconds'.format(time.time()-t))
        
        
#         # ipdb.set_trace()
#         print(time.time()-t)
#         print('Computing nearest neighbours')
#         sys.stdout.flush()
#         csls_alpha=1
#         batch_num=1
#         for i in range(0, len(src), batch_size):
#             j = min(i + batch_size, len(src))
#             similarities = xw[src[i:j]].dot(zw.T)
#             similarities = np.transpose(np.transpose(2*similarities) - csls_alpha*nbrhood_x[src[i:j]])- csls_alpha*nbrhood_z
#             nn = similarities.argmax(axis=1).tolist()
#             print(time.time()-t)
#             similarities = np.argsort((similarities),axis=1)

#             nn5 = (similarities[:,-5:])
#             nn10 = (similarities[:,-10:])
#             for k in range(j-i):
#                 translation[src[i+k]] = nn[k]
#                 translation5[src[i+k]] = nn5[k]
#                 translation10[src[i+k]] = nn10[k]
#             print('Completed batch {0} in {1}'.format(batch_num, time.time()-t))
#             sys.stdout.flush()
#             batch_num+=1
#         print('Completed in {0} seconds'.format(time.time()-t))
#         sys.stdout.flush()

#     # get translations
#     trans_pairs=[]
#     for w in trans_words: 
#         trans=''
#         if w in src_word2ind:
#             trans=tgt_words[translation[src_word2ind[w]]]
#             trans_pairs.append((w,trans))

#     return dict(trans_pairs)
   
#     ### output in moses format 
#     #with open(trans_tgt_fname, 'w', encoding='utf-8') as trans_tgt_file:
#     #    for wno, w in enumerate(trans_words): 
#     #        if w in src_word2ind:
#     #            for t_idx in reversed(translation10[src_word2ind[w]]):
#     #                trans_tgt_file.write( u'{} ||| {} ||| {} ||| {}\n'.format( 
#     #                    wno, tgt_words[t_idx], '0.0 0.0 0.0 0.0', 0.0) )
#     #        else: 
#     #            trans_tgt_file.write( u'{} ||| {} ||| {} ||| {}\n'.format( 
#     #                            wno, '@@OOV@@', '0.0 0.0 0.0 0.0', 0.0) )


def translate_topn(words_to_translate, src_emb_info, tgt_emb_info, retrieval_method="csls", topn=5, csls_k=10, batch_size=2500):
    """
    The top-n are not necessarily sorted, but the scores can be used to retrieve the sorted top-k candidates
    Only the 'csls' search implementation is complete
    """

    # Read source embeddings
    src_words, x = src_emb_info
    src_word2ind = build_w2i(src_words)
    
    # Read target embeddings
    tgt_words, z = tgt_emb_info
    tgt_word2ind = build_w2i(tgt_words)
   
    xw = embeddings.length_normalize(x)
    zw = embeddings.length_normalize(z)

    all_words=[]
    trans_words=[]
    trans_idx=[]
    oov=set()
    
    for w in words_to_translate:
        try:
            all_words.append(w)
            w_ind = src_word2ind[w]
            trans_words.append(w)
            trans_idx.append(w_ind)
        except KeyError:
            oov.add(w)
  
    print(len(all_words))
    print(len(trans_words))
    print(len(trans_idx))
    print(len(oov))
    src=trans_idx 

    translation_topn = collections.defaultdict(list)
    translation_topn_prob = collections.defaultdict(list)

    if retrieval_method == 'nn':  # Standard nearest neighbor
        for i in range(0, len(src), batch_size):
            j = min(i + batch_size, len(src))
            similarities = xw[src[i:j]].dot(zw.T)
            similarities_idx    = similarities.argsort(axis=1)
            similarities_scores = np.sort(similarities,axis=1)
            nn_topn = similarities_idx[:,-topn:]
            sim_unnorm = np.exp(similarities_scores[:,-topn:])
            sim_total =np.sum( sim_unnorm, axis=1).reshape((sim_unnorm.shape[0],1))  # sim_unnorm has same first dimension as sim_total
            nn_topn_logprob=np.log(sim_unnorm/sim_total)  ## softmax log probabilities

            for k in range(j-i):
                translation_topn[src[i+k]] = nn_topn[k]
                translation_topn_logprob[src[i+k]] = nn_topn_logprob[k]
                
    elif retrieval_method == 'csls':
        t=time.time()
        nbrhood_x=np.zeros(xw.shape[0])
        nbrhood_z=np.zeros(zw.shape[0])
        nbrhood_z2=cp.zeros(zw.shape[0])
        
        print('Computing X Neighbourhood')
        # batch_size=1000
        batch_num=1
        for i in range(0, len(src), batch_size):
            j = min(i + batch_size, len(src))
            similarities = xw[src[i:j]].dot(zw.T)
            # similarities_x = np.sort(similarities, axis=1)
            similarities_x = -1*np.partition(-1*similarities,csls_k-1 ,axis=1)
            #similarities_x = -1*cp.partition(-1*cp.dot(cp.asarray(xw[src[i:j]]),cp.transpose(cp.asarray(zw))),csls_k-1 ,axis=1)[:,:csls_k]
            nbrhood_x[src[i:j]]=np.mean(similarities_x[:,:csls_k],axis=1)
            print('Completed batch {0} in {1}'.format(batch_num, time.time()-t))
            batch_num+=1
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
            similarities = np.transpose(np.transpose(2*similarities) - \
                                csls_alpha*nbrhood_x[src[i:j]])- \
                                csls_alpha*nbrhood_z
            
            similarities_idx=-1*np.argpartition(-1*similarities,topn-1,axis=1)
            nn_topn = similarities_idx[:,-topn:]
            row_x=np.tile( np.array(range(topn))  ,
                      (similarities_idx.shape[0],1)
                     )
            print('Shapes')
            print(similarities.shape)
            print(similarities_idx.shape)
            similarities_scores=similarities[row_x,nn_topn]
            sim_unnorm = np.exp(similarities_scores)                                              
            
#             similarities_idx = similarities.argsort(axis=1)
#             similarities_scores = np.sort(similarities,axis=1)
#             print(time.time()-t)
#             nn_topn = similarities_idx[:,-topn:]
#             sim_unnorm = np.exp(similarities_scores[:,-topn:])                                              
                                              
            sim_total =np.sum( sim_unnorm, axis=1).reshape((sim_unnorm.shape[0],1))  # sim_unnorm has same first dimension as sim_total
#             nn_topn_logprob=np.log(sim_unnorm/sim_total)  ## softmax log probabilities
            nn_topn_prob=sim_unnorm/sim_total  ## softmax log probabilities

            for k in range(j-i):
                translation_topn[src[i+k]] = nn_topn[k]
                translation_topn_prob[src[i+k]] = nn_topn_prob[k]

        print('Completed in {0} seconds'.format(time.time()-t))

    # get translations
    trans_pairs=[]
    for w in trans_words: 
        if w in src_word2ind:
            srcid=src_word2ind[w]
            trans=[ (tgt_words[translation_topn[srcid][r]], translation_topn_prob[srcid][r]) for r in range(topn) ]
            trans_pairs.append((w,trans))

    return dict(trans_pairs)
   
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

def compute_prefix_embeddings(words, emb_info, dtype='float'):
    """
    - Words in the pre-trained embeddings: discarded in the output
    - Words not in the pre-trained embeddings: the embeddings are computed as mean of the embeddings
    of words which share prefixes with the input words.
    - Words with no matching prefix: discarded in the output
    """
    
    emb_words, emb = emb_info
    emb_w2i=build_w2i(emb_words)
    
    emb_words_trie = pygtrie.CharTrie()
    for w in emb_words:
        emb_words_trie[w] = 1
    
    output_pairs=[]
    for w in words:
        if  w not in emb_w2i:
#             print("===" + w)
            ## handle emb words for which the input word is a prefix
            longer_words=[]
            if emb_words_trie.has_key(w):
                longer_words=emb_words_trie.keys(w)
            
            ## handle emb words which are prefixes of the input word is a prefix
            shorter_words=[ x[0] for x in emb_words_trie.prefixes(w) ]
            
#             ## all matched words 
#             matched_words=longer_words+shorter_words
            
            ## longest short word and shortest long word
            matched_words = []
            if len(longer_words)>0:
                matched_words.append( min(longer_words,key=lambda x:len(x)) )
            if len(shorter_words)>0:
                matched_words.append( max(shorter_words,key=lambda x:len(x)) )

#             print(matched_words)
                
            ## embedding of word is mean of matched words embeddings 
            if len(matched_words)>0:
                w_emb=np.mean(
                            np.array(
                                    [ emb[emb_w2i[mw]] for mw in matched_words ], 
                            dtype=dtype),
                        axis=0
                    )
                output_pairs.append((w,w_emb))
    
    output_words=[ x[0] for x in output_pairs]
    output_emb = np.array([ x[1] for x in output_pairs], dtype=dtype)
    
    return (output_words, output_emb)

def read_word_similarity(similarity_fname, delim='\t'):
    sim_database=[]
    with open(similarity_fname,'r',encoding='utf-8') as similarity_file:
        for l in similarity_file:
            f      = l.strip().split(delim)
            sim_database.append((f[0],f[1],float(f[2])))
    return sim_database

def read_bilingual_dict(fname,delim=' '):
    trans_dict=collections.defaultdict(set)
    with open(fname,'r',encoding='utf-8') as infile:
        for i,line in enumerate(infile,1):
            r=line.strip().split(delim)
            if len(r)!=2:
                print('Ignore entry on line {}: does not have 2 fields'.format(i))
                continue
            k,v=r
            trans_dict[k].add(v)

    return trans_dict

def compute_word_similarity(emb_info, sim_database):
            
    emb_words, emb_vectors = emb_info
    w2i=build_w2i(emb_info[0])
    
    sim_words = set([ x[0] for x in sim_database ])
    sim_words.update([ x[1] for x in sim_database ])
    oov_words = sim_words.difference(emb_words)
    non_oov_words=sim_words.difference(oov_words)
    
    non_oov_sim_pairs = list(filter( lambda x: len(oov_words.intersection(x[:2]))==0 , sim_database))

    cos_sims=[]
    ref_sims=[]
    
    for w1, w2, ref_sim in non_oov_sim_pairs:
        v1=emb_vectors[w2i[w1]]
        v2=emb_vectors[w2i[w2]]
        cos_sim=np.dot(v1,v2)/np.sqrt(v1.dot(v1)*v2.dot(v2))
        
        cos_sims.append(cos_sim)
        ref_sims.append(ref_sim)
    
    corr=scipy.stats.spearmanr(np.array(cos_sims),np.array(ref_sims))
    return corr[0], corr[1], len(non_oov_sim_pairs)/len(sim_database)

def get_oov_info_for_bilingual_dict(train_dict_fname, test_dict_fname, 
                       src_emb_fname, tgt_emb_fname,max_voc=200000):

    
    ## read dictionaries
    train_dict = read_bilingual_dict(train_dict_fname)    
    test_dict  = read_bilingual_dict(test_dict_fname)
    
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
    for l in train_dict.values():
        tgt_oov_words.update(l)
    for l in test_dict.values():
        tgt_oov_words.update(l)
    tgt_oov_words.difference_update(tgt_vcb_words) 
    print('Number of tgt OOV words: {}'.format(len(tgt_oov_words)))
    
    return (src_oov_words, (src_vcb_words, src_emb), tgt_oov_words, (tgt_vcb_words, tgt_emb))

def ft_oov_embeddings_for_bilingual_dict(train_dict_fname, test_dict_fname, 
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
    
    src_oov_words, src_emb_info, tgt_oov_words, tgt_emb_info = \
        get_oov_info_for_bilingual_dict(train_dict_fname, test_dict_fname, 
                       src_emb_fname, tgt_emb_fname, max_voc)
    
    src_vcb_words, src_emb = src_emb_info
    tgt_vcb_words, tgt_emb = tgt_emb_info
    
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
    
def prefix_oov_embeddings_for_bilingual_dict(train_dict_fname, test_dict_fname, 
                       src_emb_fname, tgt_emb_fname,
                       out_src_emb_fname, out_tgt_emb_fname,
                       max_voc=200000):
    """
    Adds the embeddings for OOV words in the training and test dictionaries to the embedding file. 
    This is done by using prefix of the word as well as words for which oov is a prefix. 
    Note that the output embedding file will contain only the OOV words plus 
    the first max_voc words in the original embedding file.
    
    train_dict_fname: 
    test_dict_fname: 
    src_emb_fname: embedding file for source language 
    tgt_emb_fname: embedding file for target language
    out_src_emb_fname: output embedding file for source language 
    out_tgt_emb_fname: output embedding file for target language    
    max_voc: number of vocab items to process from the embedding file

    """
    
    src_oov_words, src_emb_info, tgt_oov_words, tgt_emb_info = \
        get_oov_info_for_bilingual_dict(train_dict_fname, test_dict_fname, 
                       src_emb_fname, tgt_emb_fname, max_voc)
    
    src_vcb_words, src_emb = src_emb_info
    tgt_vcb_words, tgt_emb = tgt_emb_info
    
    ## compute embeddings for OOV
    ##### cat queries.txt | ./fasttext print-word-vectors model.bin
    src_oov_final_words, src_oov_emb = compute_prefix_embeddings(src_oov_words, (src_vcb_words, src_emb))
    tgt_oov_final_words, tgt_oov_emb = compute_prefix_embeddings(tgt_oov_words, (tgt_vcb_words, tgt_emb))
    
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
        
        
def filter_embeddings(in_embfname,filter_func):        
    
    embeddings.read(in_embfile,max_voc=max_voc)