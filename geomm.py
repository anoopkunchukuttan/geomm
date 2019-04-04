# Code for GeoMM algorithm

import embeddings
import argparse
import collections
import numpy as np
import cupy as cp
import scipy.linalg
import sys
import ipdb 
import time
import os
import theano.tensor as TT
from theano import shared
import datetime
from pymanopt import Problem
from pymanopt.manifolds import Stiefel, Product, PositiveDefinite, Euclidean
from pymanopt.solvers import ConjugateGradient, TrustRegions
import gc

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Map the source embeddings into the target embedding space')
    parser.add_argument('src_input', help='the input source embeddings')
    parser.add_argument('trg_input', help='the input target embeddings')
    parser.add_argument('--model_path', default=None, type=str, help='directory to save the model')
    parser.add_argument('--geomm_embeddings_path', default=None, type=str, help='directory to save the output GeoMM latent space embeddings. The output embeddings are normalized.')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--max_vocab', default=0,type=int, help='Maximum vocabulary to be loaded, 0 allows complete vocabulary')
    parser.add_argument('--verbose', default=0,type=int, help='Verbose')
    mapping_group = parser.add_argument_group('mapping arguments', 'Basic embedding mapping arguments')
    mapping_group.add_argument('-dtrain', '--dictionary_train', default=sys.stdin.fileno(), help='the training dictionary file (defaults to stdin)')
    mapping_group.add_argument('-dtest', '--dictionary_test', default=sys.stdin.fileno(), help='the test dictionary file (defaults to stdin)')
    mapping_group.add_argument('--normalize', choices=['unit', 'center', 'unitdim', 'centeremb'], nargs='*', default=[], help='the normalization actions to perform in order')
    
    geomm_group = parser.add_argument_group('GeoMM arguments', 'Arguments for GeoMM method')
    geomm_group.add_argument('--l2_reg', type=float,default=1e2, help='Lambda for L2 Regularization')
    geomm_group.add_argument('--max_opt_time', type=int,default=5000, help='Maximum time limit for optimization in seconds')
    geomm_group.add_argument('--max_opt_iter', type=int,default=150, help='Maximum number of iterations for optimization')
   
    eval_group = parser.add_argument_group('evaluation arguments', 'Arguments for evaluation')
    eval_group.add_argument('--normalize_eval', action='store_true', help='Normalize the embeddings at test time')
    eval_group.add_argument('--eval_batch_size', type=int,default=1000, help='Batch size for evaluation')
    eval_group.add_argument('--csls_neighbourhood', type=int,default=10, help='Neighbourhood size for CSLS')

    args = parser.parse_args()
    BATCH_SIZE = args.eval_batch_size
    
    # Logging
    method_name = os.path.join('logs','geomm')
    directory = os.path.join(os.path.join(os.getcwd(),method_name), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(directory):
        os.makedirs(directory)
    log_file_name, file_extension = os.path.splitext(os.path.basename(args.dictionary_train))
    log_file_name = log_file_name + '.log'
    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(os.path.join(directory,log_file_name), "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)  

        def flush(self):
            #this flush method is needed for python 3 compatibility.
            #this handles the flush command by doing nothing.
            #you might want to specify some extra behavior here.
            pass    
    sys.stdout = Logger()
    if args.verbose:
        print('Current arguments: {0}'.format(args))

    dtype = 'float32'
    if args.verbose:
        print('Loading train data...')
    # Read input embeddings
    srcfile = open(args.src_input, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_input, encoding=args.encoding, errors='surrogateescape')
    src_words, x = embeddings.read(srcfile,max_voc=args.max_vocab, dtype=dtype)
    trg_words, z = embeddings.read(trgfile,max_voc=args.max_vocab, dtype=dtype)

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    # Build training dictionary
    src_indices = []
    trg_indices = []
    f = open(args.dictionary_train, encoding=args.encoding, errors='surrogateescape')
    for line in f:
        src,trg = line.split()
        if args.max_vocab:
            src=src.lower()
            trg=trg.lower()
        try:
            src_ind = src_word2ind[src]
            trg_ind = trg_word2ind[trg]
            src_indices.append(src_ind)
            trg_indices.append(trg_ind)
        except KeyError:
            if args.verbose:
                print('WARNING: OOV dictionary entry ({0} - {1})'.format(src, trg), file=sys.stderr)
    f.close()
    src_indices = src_indices
    trg_indices = trg_indices
    if args.verbose:
        print('Normalizing embeddings...')
    # STEP 0: Normalization
    for action in args.normalize:
        if action == 'unit':
            x = embeddings.length_normalize(x)
            z = embeddings.length_normalize(z)
        elif action == 'center':
            x = embeddings.mean_center(x)
            z = embeddings.mean_center(z)
        elif action == 'unitdim':
            x = embeddings.length_normalize_dimensionwise(x)
            z = embeddings.length_normalize_dimensionwise(z)
        elif action == 'centeremb':
            x = embeddings.mean_center_embeddingwise(x)
            z = embeddings.mean_center_embeddingwise(z)


    # Step 1: Optimization
    if args.verbose:
        print('Beginning Optimization')
    start_time = time.time()
    x_count = len(set(src_indices))
    z_count = len(set(trg_indices))
    A = np.zeros((x_count,z_count))
    
    # Creating dictionary matrix from training set
    map_dict_src={}
    map_dict_trg={}
    I=0
    uniq_src=[]
    uniq_trg=[]
    for i in range(len(src_indices)):
        if src_indices[i] not in map_dict_src.keys():
            map_dict_src[src_indices[i]]=I
            I+=1
            uniq_src.append(src_indices[i])
    J=0
    for j in range(len(trg_indices)):
        if trg_indices[j] not in map_dict_trg.keys():
            map_dict_trg[trg_indices[j]]=J
            J+=1
            uniq_trg.append(trg_indices[j])

    for i in range(len(src_indices)):
        A[map_dict_src[src_indices[i]],map_dict_trg[trg_indices[i]]]=1

    np.random.seed(0)
    Lambda=args.l2_reg
    
    U1 = TT.matrix()
    U2 = TT.matrix()
    B  = TT.matrix()

    cost = TT.sum(((shared(x[uniq_src]).dot(U1.dot(B.dot(U2.T)))).dot(shared(z[uniq_trg]).T)-A)**2) + 0.5*Lambda*(TT.sum(B**2))

    solver = ConjugateGradient(maxtime=args.max_opt_time,maxiter=args.max_opt_iter)

    manifold =Product([Stiefel(x.shape[1], x.shape[1]),Stiefel(z.shape[1], x.shape[1]),PositiveDefinite(x.shape[1])])
    problem = Problem(manifold=manifold, cost=cost, arg=[U1,U2,B], verbosity=3)
    wopt = solver.solve(problem)

    w= wopt
    U1 = w[0]
    U2 = w[1]
    B = w[2]

    ### Save the models if requested
    if args.model_path is not None: 
        os.makedirs(args.model_path,exist_ok=True)
        np.savetxt('{}/U_src.csv'.format(args.model_path),U1)
        np.savetxt('{}/U_tgt.csv'.format(args.model_path),U2)
        np.savetxt('{}/B.csv'.format(args.model_path),B)

    # Step 2: Transformation
    xw = x.dot(U1).dot(scipy.linalg.sqrtm(B))
    zw = z.dot(U2).dot(scipy.linalg.sqrtm(B))

    end_time = time.time()
    if args.verbose:
        print('Completed training in {0:.2f} seconds'.format(end_time-start_time))
    gc.collect()

    ### Save the GeoMM embeddings if requested
    xw_n = embeddings.length_normalize(xw)
    zw_n = embeddings.length_normalize(zw)
    if args.geomm_embeddings_path is not None: 
        os.makedirs(args.geomm_embeddings_path,exist_ok=True)

        out_emb_fname=os.path.join(args.geomm_embeddings_path,'src.vec')
        with open(out_emb_fname,'w',encoding=args.encoding) as outfile:
            embeddings.write(src_words,xw_n,outfile)

        out_emb_fname=os.path.join(args.geomm_embeddings_path,'trg.vec')
        with open(out_emb_fname,'w',encoding=args.encoding) as outfile:
            embeddings.write(trg_words,zw_n,outfile)

    # Step 3: Evaluation
    if args.normalize_eval:
        xw = xw_n
        zw = zw_n

    X = xw[src_indices]
    Z = zw[trg_indices]

    # Loading test dictionary
    f = open(args.dictionary_test, encoding=args.encoding, errors='surrogateescape')
    src2trg = collections.defaultdict(set)
    trg2src = collections.defaultdict(set)
    oov = set()
    vocab = set()
    for line in f:
        src, trg = line.split()
        if args.max_vocab:
            src=src.lower()
            trg=trg.lower()
        try:
            src_ind = src_word2ind[src]
            trg_ind = trg_word2ind[trg]
            src2trg[src_ind].add(trg_ind)
            trg2src[trg_ind].add(src_ind)
            vocab.add(src)
        except KeyError:
            oov.add(src)
    src = list(src2trg.keys())
    trgt = list(trg2src.keys())

    oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
    coverage = len(src2trg) / (len(src2trg) + len(oov))
    f.close()

    translation = collections.defaultdict(int)
    translation5 = collections.defaultdict(list)
    translation10 = collections.defaultdict(list)

    ### compute nearest neigbours of x in z
    t=time.time()
    nbrhood_x=np.zeros(xw.shape[0])

    for i in range(0, len(src), BATCH_SIZE):
        j = min(i + BATCH_SIZE, len(src))
        similarities = xw[src[i:j]].dot(zw.T)
        similarities_x = -1*np.partition(-1*similarities,args.csls_neighbourhood-1 ,axis=1)
        nbrhood_x[src[i:j]]=np.mean(similarities_x[:,:args.csls_neighbourhood],axis=1)

    ### compute nearest neigbours of z in x (GPU version)
    nbrhood_z=np.zeros(zw.shape[0])
    nbrhood_z2=cp.zeros(zw.shape[0])
    batch_num=1
    for i in range(0, zw.shape[0], BATCH_SIZE):
        j = min(i + BATCH_SIZE, zw.shape[0])
        similarities = -1*cp.partition(-1*cp.dot(cp.asarray(zw[i:j]),cp.transpose(cp.asarray(xw))),args.csls_neighbourhood-1 ,axis=1)[:,:args.csls_neighbourhood]
        nbrhood_z2[i:j]=(cp.mean(similarities[:,:args.csls_neighbourhood],axis=1))
        batch_num+=1
    nbrhood_z=cp.asnumpy(nbrhood_z2)

    #### compute nearest neigbours of z in x (CPU version)
    #nbrhood_z=np.zeros(zw.shape[0])
    #for i in range(0, len(zw.shape[0]), BATCH_SIZE):
    #    j = min(i + BATCH_SIZE, len(zw.shape[0]))
    #    similarities = zw[i:j].dot(xw.T)
    #    similarities_z = -1*np.partition(-1*similarities,args.csls_neighbourhood-1 ,axis=1)
    #    nbrhood_z[i:j]=np.mean(similarities_z[:,:args.csls_neighbourhood],axis=1)

    ### find translation 
    for i in range(0, len(src), BATCH_SIZE):
        j = min(i + BATCH_SIZE, len(src))
        similarities = xw[src[i:j]].dot(zw.T)
        similarities = np.transpose(np.transpose(2*similarities) - nbrhood_x[src[i:j]]) - nbrhood_z
        nn = similarities.argmax(axis=1).tolist()
        similarities = np.argsort((similarities),axis=1)

        nn5 = (similarities[:,-5:])
        nn10 = (similarities[:,-10:])
        for k in range(j-i):
            translation[src[i+k]] = nn[k]
            translation5[src[i+k]] = nn5[k]
            translation10[src[i+k]] = nn10[k]
    accuracy = np.mean([1 if translation[i] in src2trg[i] else 0 for i in src])
    mean=0
    for i in src:
        for k in translation5[i]:
            if k in src2trg[i]:
                mean+=1
                break

    mean/=len(src)
    accuracy5 = mean

    mean=0
    for i in src:
        for k in translation10[i]:
            if k in src2trg[i]:
                mean+=1
                break

    mean/=len(src)
    accuracy10 = mean
    print('Coverage:{0:7.2%}  Accuracy:{1:7.2%}  Accuracy(Top 5):{2:7.2%}  Accuracy(Top 10):{3:7.2%}'.format(coverage, accuracy, accuracy5, accuracy10))

if __name__ == '__main__':
    main()
