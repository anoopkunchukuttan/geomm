# Code for GeoMM used in Composition and Pipeline methods for Trilingual experiments.

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
    parser.add_argument('mid_input', help='the input pivot embeddings')
    parser.add_argument('trg_input', help='the input target embeddings')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--max_vocab', default=0,type=int, help='Maximum vocabulary to be loaded, 0 allows complete vocabulary')
    parser.add_argument('--verbose', default=0,type=int, help='Verbose')  
    mapping_group = parser.add_argument_group('mapping arguments', 'Basic embedding mapping arguments')
    mapping_group.add_argument('-dtrain1', '--dictionary_train1', default=sys.stdin.fileno(), help='the first training dictionary file (defaults to stdin)')
    mapping_group.add_argument('-dtrain2', '--dictionary_train2', default=sys.stdin.fileno(), help='the second training dictionary file (defaults to stdin)')
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
    method_name = os.path.join('logs','geomm_cmp_pip')
    directory = os.path.join(os.path.join(os.getcwd(),method_name), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(directory):
        os.makedirs(directory)
    log_file_name, file_extension = os.path.splitext(os.path.basename(args.dictionary_test))
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
    midfile = open(args.mid_input, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_input, encoding=args.encoding, errors='surrogateescape')
    
    src_words, x = embeddings.read(srcfile,max_voc=args.max_vocab, dtype=dtype)
    mid_words, y = embeddings.read(midfile,max_voc=args.max_vocab, dtype=dtype)
    trg_words, z = embeddings.read(trgfile,max_voc=args.max_vocab, dtype=dtype)

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    mid_word2ind = {word: i for i, word in enumerate(mid_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    # Build training dictionary-1
    src_indices12 = []
    trg_indices12 = []
    f = open(args.dictionary_train1, encoding=args.encoding, errors='surrogateescape')
    for line in f:
        src,trg = line.split()
        if args.max_vocab:
            src=src.lower()
            trg=trg.lower()
        try:
            src_ind = src_word2ind[src]
            trg_ind = mid_word2ind[trg]
            src_indices12.append(src_ind)
            trg_indices12.append(trg_ind)
        except KeyError:
            if args.verbose:
                print('WARNING: OOV dictionary entry ({0} - {1})'.format(src, trg), file=sys.stderr)
    f.close()

    # Build training dictionary-2
    src_indices23= []
    trg_indices23 = []
    f = open(args.dictionary_train2, encoding=args.encoding, errors='surrogateescape')
    for line in f:
        src,trg = line.split()
        if args.max_vocab:
            src=src.lower()
            trg=trg.lower()
        try:
            src_ind = mid_word2ind[src]
            trg_ind = trg_word2ind[trg]
            src_indices23.append(src_ind)
            trg_indices23.append(trg_ind)
        except KeyError:
            if args.verbose:
                print('WARNING: OOV dictionary entry ({0} - {1})'.format(src, trg), file=sys.stderr)
    f.close()

    if args.verbose:
        print('Normalizing embeddings...')
    # STEP 0: Normalization
    for action in args.normalize:
        if action == 'unit':
            x = embeddings.length_normalize(x)
            y = embeddings.length_normalize(y)
            z = embeddings.length_normalize(z)
        elif action == 'center':
            x = embeddings.mean_center(x)
            y = embeddings.mean_center(y)
            z = embeddings.mean_center(z)
        elif action == 'unitdim':
            x = embeddings.length_normalize_dimensionwise(x)
            y = embeddings.length_normalize_dimensionwise(y)
            z = embeddings.length_normalize_dimensionwise(z)
        elif action == 'centeremb':
            x = embeddings.mean_center_embeddingwise(x)
            y = embeddings.mean_center_embeddingwise(y)
            z = embeddings.mean_center_embeddingwise(z)

    # Step 1.1: Optimization-1
    if args.verbose:
        print('Beginning Optimization-1')
    start_time = time.time()
    
    x_count = len(set(src_indices12))
    y_count = len(set(trg_indices12))
    A = np.zeros((x_count,y_count))
    
    # Creating dictionary matrix from training set
    map_dict_src={}
    map_dict_trg={}
    I=0
    uniq_src=[]
    uniq_trg=[]
    for i in range(len(src_indices12)):
        if src_indices12[i] not in map_dict_src.keys():
            map_dict_src[src_indices12[i]]=I
            I+=1
            uniq_src.append(src_indices12[i])
    J=0
    for j in range(len(trg_indices12)):
        if trg_indices12[j] not in map_dict_trg.keys():
            map_dict_trg[trg_indices12[j]]=J
            J+=1
            uniq_trg.append(trg_indices12[j])

    for i in range(len(src_indices12)):
        A[map_dict_src[src_indices12[i]],map_dict_trg[trg_indices12[i]]]=1

    np.random.seed(0)
    Lambda=args.l2_reg
    U1 = TT.matrix()
    U2 = TT.matrix()
    B = TT.matrix()
    cost = TT.sum(((shared(x[uniq_src]).dot(U1.dot(B.dot(U2.T)))).dot(shared(y[uniq_trg]).T)-A)**2) + 0.5*Lambda*(TT.sum(B**2))

    solver = ConjugateGradient(maxtime=args.max_opt_time,maxiter=args.max_opt_iter)

    low_rank=300
    manifold =Product([Stiefel(x.shape[1], low_rank),Stiefel(y.shape[1], low_rank),PositiveDefinite(low_rank)])
    problem = Problem(manifold=manifold, cost=cost, arg=[U1,U2,B], verbosity=3)
    wopt = solver.solve(problem)

    w= wopt
    U1 = w[0]
    U2 = w[1]
    B = w[2]
    w12 = U1.dot(B).dot(U2.T)
    u11=U1
    u21=U2
    b1=B
    
    # Step 1.2: Optimization-2
    if args.verbose:
        print('Beginning Optimization-2')
    y_count = len(set(src_indices23))
    z_count = len(set(trg_indices23))
    A = np.zeros((y_count,z_count))
    
    # Creating dictionary matrix from training set
    map_dict_src={}
    map_dict_trg={}
    I=0
    uniq_src=[]
    uniq_trg=[]
    for i in range(len(src_indices23)):
        if src_indices23[i] not in map_dict_src.keys():
            map_dict_src[src_indices23[i]]=I
            I+=1
            uniq_src.append(src_indices23[i])
    J=0
    for j in range(len(trg_indices23)):
        if trg_indices23[j] not in map_dict_trg.keys():
            map_dict_trg[trg_indices23[j]]=J
            J+=1
            uniq_trg.append(trg_indices23[j])

    for i in range(len(src_indices23)):
        A[map_dict_src[src_indices23[i]],map_dict_trg[trg_indices23[i]]]=1

    np.random.seed(0)
    U1 = TT.matrix()
    U2 = TT.matrix()
    B = TT.matrix()
    cost = TT.sum(((shared(y[uniq_src]).dot(U1.dot(B.dot(U2.T)))).dot(shared(z[uniq_trg]).T)-A)**2) + 0.5*Lambda*(TT.sum(B**2))
    solver = ConjugateGradient(maxtime=args.max_opt_time,maxiter=args.max_opt_iter)

    low_rank=300
    manifold =Product([Stiefel(y.shape[1], low_rank),Stiefel(z.shape[1], low_rank),PositiveDefinite(low_rank)])
    problem = Problem(manifold=manifold, cost=cost, arg=[U1,U2,B], verbosity=3)
    wopt = solver.solve(problem)

    w= wopt
    U1 = w[0]
    U2 = w[1]
    B = w[2]
    w23 = U1.dot(B).dot(U2.T)
    u22=U1
    u32=U2
    b2=B

    # Step 2: Transformation
    w12_1=u11.dot(scipy.linalg.sqrtm(b1))
    w12_2=u21.dot(scipy.linalg.sqrtm(b1))
    w23_1=u22.dot(scipy.linalg.sqrtm(b2))
    w23_2=u32.dot(scipy.linalg.sqrtm(b2))

    end_time = time.time()
    if args.verbose:
        print('Completed training in {0:.2f} seconds'.format(end_time-start_time))
    gc.collect()

    # Step 3: Evaluation
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

    # Composition (CMP)
    xw = x.dot(w12).dot(w23)
    zw = z
    if args.normalize_eval:
        xw = embeddings.length_normalize(xw)
        zw = embeddings.length_normalize(zw)



    translation = collections.defaultdict(int)
    translation5 = collections.defaultdict(list)
    translation10 = collections.defaultdict(list)

    t=time.time()
    nbrhood_x=np.zeros(xw.shape[0])
    nbrhood_z=np.zeros(zw.shape[0])
    nbrhood_z2=cp.zeros(zw.shape[0])
    for i in range(0, len(src), BATCH_SIZE):
        j = min(i + BATCH_SIZE, len(src))
        similarities = xw[src[i:j]].dot(zw.T)
        similarities_x = -1*np.partition(-1*similarities,args.csls_neighbourhood-1 ,axis=1)
        nbrhood_x[src[i:j]]=np.mean(similarities_x[:,:args.csls_neighbourhood],axis=1)

    batch_num=1
    for i in range(0, zw.shape[0], BATCH_SIZE):
        j = min(i + BATCH_SIZE, zw.shape[0])
        similarities = -1*cp.partition(-1*cp.dot(cp.asarray(zw[i:j]),cp.transpose(cp.asarray(xw))),args.csls_neighbourhood-1 ,axis=1)[:,:args.csls_neighbourhood]
        nbrhood_z2[i:j]=(cp.mean(similarities[:,:args.csls_neighbourhood],axis=1))
        batch_num+=1
    nbrhood_z=cp.asnumpy(nbrhood_z2)
    for i in range(0, len(src), BATCH_SIZE):
        j = min(i + BATCH_SIZE, len(src))
        similarities = xw[src[i:j]].dot(zw.T)
        similarities = np.transpose(np.transpose(2*similarities) - nbrhood_x[src[i:j]])- nbrhood_z
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
    print('CMP: Coverage:{0:7.2%}  Accuracy:{1:7.2%}  Accuracy(Top 5):{2:7.2%}  Accuracy(Top 10):{3:7.2%}'.format(coverage, accuracy, accuracy5, accuracy10))



    # Pipeline (PIP)
    xw = x.dot(w12_1)
    zw = y.dot(w12_2)
    if args.normalize_eval:
        xw = embeddings.length_normalize(xw)
        zw = embeddings.length_normalize(zw)

    translation12 = collections.defaultdict(int)
    # PIP-Stage 1
    t=time.time()
    nbrhood_x=np.zeros(xw.shape[0])
    nbrhood_z=np.zeros(zw.shape[0])
    nbrhood_z2=cp.zeros(zw.shape[0])
    for i in range(0, len(src), BATCH_SIZE):
        j = min(i + BATCH_SIZE, len(src))
        similarities = xw[src[i:j]].dot(zw.T)
        similarities_x = -1*np.partition(-1*similarities,args.csls_neighbourhood-1 ,axis=1)
        nbrhood_x[src[i:j]]=np.mean(similarities_x[:,:args.csls_neighbourhood],axis=1)

    batch_num=1
    for i in range(0, zw.shape[0], BATCH_SIZE):
        j = min(i + BATCH_SIZE, zw.shape[0])
        similarities = -1*cp.partition(-1*cp.dot(cp.asarray(zw[i:j]),cp.transpose(cp.asarray(xw))),args.csls_neighbourhood-1 ,axis=1)[:,:args.csls_neighbourhood]
        nbrhood_z2[i:j]=(cp.mean(similarities[:,:args.csls_neighbourhood],axis=1))
        batch_num+=1
    nbrhood_z=cp.asnumpy(nbrhood_z2)
    for i in range(0, len(src), BATCH_SIZE):
        j = min(i + BATCH_SIZE, len(src))
        similarities = xw[src[i:j]].dot(zw.T)
        similarities = np.transpose(np.transpose(2*similarities) - nbrhood_x[src[i:j]])- nbrhood_z
        nn = similarities.argmax(axis=1).tolist()
        for k in range(j-i):
            translation[src[i+k]] = nn[k]

    # PIP-Stage 2
    mid = [translation[sr] for sr in src]
    xw = y.dot(w23_1)
    zw = z.dot(w23_2)
    if args.normalize_eval:
        xw = embeddings.length_normalize(xw)
        zw = embeddings.length_normalize(zw)

    translation = collections.defaultdict(int)
    translation5 = collections.defaultdict(list)
    translation10 = collections.defaultdict(list)

    t=time.time()
    nbrhood_x=np.zeros(xw.shape[0])
    nbrhood_z=np.zeros(zw.shape[0])
    nbrhood_z2=cp.zeros(zw.shape[0])
    for i in range(0, len(mid), BATCH_SIZE):
        j = min(i + BATCH_SIZE, len(mid))
        similarities = xw[mid[i:j]].dot(zw.T)
        # similarities_x = np.sort(similarities, axis=1)
        similarities_x = -1*np.partition(-1*similarities,args.csls_neighbourhood-1 ,axis=1)
        nbrhood_x[mid[i:j]]=np.mean(similarities_x[:,:args.csls_neighbourhood],axis=1)

    batch_num=1
    for i in range(0, zw.shape[0], BATCH_SIZE):
        j = min(i + BATCH_SIZE, zw.shape[0])
        similarities = -1*cp.partition(-1*cp.dot(cp.asarray(zw[i:j]),cp.transpose(cp.asarray(xw))),args.csls_neighbourhood-1 ,axis=1)[:,:args.csls_neighbourhood]
        nbrhood_z2[i:j]=(cp.mean(similarities[:,:args.csls_neighbourhood],axis=1))
        batch_num+=1
    nbrhood_z=cp.asnumpy(nbrhood_z2)
    for i in range(0, len(mid), BATCH_SIZE):
        j = min(i + BATCH_SIZE, len(mid))
        similarities = xw[mid[i:j]].dot(zw.T)
        similarities = np.transpose(np.transpose(2*similarities) - nbrhood_x[mid[i:j]])- nbrhood_z
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
    print('PIP: Coverage:{0:7.2%}  Accuracy:{1:7.2%}  Accuracy(Top 5):{2:7.2%}  Accuracy(Top 10):{3:7.2%}'.format(coverage, accuracy, accuracy5, accuracy10))




if __name__ == '__main__':
    main()