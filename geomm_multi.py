# Code for GeoMM-Multi algorithm

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
from pymanopt.manifolds import Stiefel, Product, PositiveDefinite
from pymanopt.solvers import ConjugateGradient, TrustRegions #, ConjugateGradientMS
import gc

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Map the source embeddings into the target embedding space')
    parser.add_argument('emb_file', help='the input target embeddings')
    parser.add_argument('--lang_list', default='', help='the list of languages listed in the same order as in the input embedding `emb_file` (comma-separated). e.g. "en,es,fr"')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--model_path', default=None, type=str, help='directory to save the model')
    parser.add_argument('--geomm_embeddings_path', default=None, type=str, help='directory to save the output GeoMM Multi latent space embeddings. The output embeddings are normalized.')

    parser.add_argument('--max_vocab', default=0,type=int, help='Maximum vocabulary to be loaded, 0 allows complete vocabulary')
    parser.add_argument('--verbose', default=0,type=int, help='Verbose')
  
    mapping_group = parser.add_argument_group('mapping arguments', 'Basic embedding mapping arguments')
    mapping_group.add_argument('-dtrain_file', '--dictionary_train_file', default=sys.stdin.fileno(), help='the training dictionary file (defaults to stdin)')
    mapping_group.add_argument('-dtest_file', '--dictionary_test_file', default=sys.stdin.fileno(), help='the test dictionary file (defaults to stdin)')
    mapping_group.add_argument('--normalize', choices=['unit', 'center', 'unitdim', 'centeremb'], nargs='*', default=[], help='the normalization actions to perform in order')
    
    geomm_group = parser.add_argument_group('GeoMM Multi arguments', 'Arguments for GeoMM Multi method')
    geomm_group.add_argument('--l2_reg', type=float,default=1e3, help='Lambda for L2 Regularization')
    geomm_group.add_argument('--max_opt_time', type=int,default=5000, help='Maximum time limit for optimization in seconds')
    geomm_group.add_argument('--max_opt_iter', type=int,default=150, help='Maximum number of iterations for optimization')
   
    eval_group = parser.add_argument_group('evaluation arguments', 'Arguments for evaluation')
    eval_group.add_argument('--normalize_eval', action='store_true', help='Normalize the embeddings at test time')
    eval_group.add_argument('--eval_batch_size', type=int,default=1000, help='Batch size for evaluation')
    eval_group.add_argument('--csls_neighbourhood', type=int,default=10, help='Neighbourhood size for CSLS')

    args = parser.parse_args()

    BATCH_SIZE = args.eval_batch_size
    lang_list=None

    ## Logging
    #method_name = os.path.join('logs','geomm_multi')
    #directory = os.path.join(os.path.join(os.getcwd(),method_name), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    #if not os.path.exists(directory):
    #    os.makedirs(directory)
    #log_file_name, file_extension = os.path.splitext(os.path.basename(args.dictionary_train_file))
    #log_file_name = log_file_name + '.log'
    #class Logger(object):
    #    def __init__(self):
    #        self.terminal = sys.stdout
    #        self.log = open(os.path.join(directory,log_file_name), "a")

    #    def write(self, message):
    #        self.terminal.write(message)
    #        self.log.write(message)  

    #    def flush(self):
    #        #this flush method is needed for python 3 compatibility.
    #        #this handles the flush command by doing nothing.
    #        #you might want to specify some extra behavior here.
    #        pass    
    #sys.stdout = Logger()
    if args.verbose:
        print('Current arguments: {0}'.format(args))

    dtype = 'float32'

    if args.verbose:
        print('Loading train data...')
    words = []
    emb = []
    with open(args.emb_file, encoding=args.encoding, errors='surrogateescape') as f:
        for line in f:
            srcfile = open(line.strip(), encoding=args.encoding, errors='surrogateescape')
            words_temp, x_temp = embeddings.read(srcfile,max_voc=args.max_vocab, dtype=dtype)
            words.append(words_temp)
            emb.append(x_temp)

    # Build word to index map
    word2ind = []
    for lang in words:
        word2ind.append({word: i for i, word in enumerate(lang)})

    ##### Set language names

    ## language id map
    if args.lang_list=='':
        lang_list = [ str(i) for i in range(len(emb)) ]
    else:
        lang_list = args.lang_list.split(',')


    # Build training dictionary
    train_pairs = []
    with open(args.dictionary_train_file, encoding=args.encoding, errors='surrogateescape') as ff:
        for line in ff:
            vals = line.split(',')
            curr_dict=[int(vals[0].strip()),int(vals[1].strip())]
            src_indices = []
            trg_indices = []
            with open(vals[2].strip(), encoding=args.encoding, errors='surrogateescape') as f:
                for line in f:
                    src,trg = line.split()
                    if args.max_vocab:
                        src=src.lower()
                        trg=trg.lower()
                    try:
                        src_ind = word2ind[curr_dict[0]][src]
                        trg_ind = word2ind[curr_dict[1]][trg]
                        src_indices.append(src_ind)
                        trg_indices.append(trg_ind)
                    except KeyError:
                        if args.verbose:
                            print('WARNING: OOV dictionary entry ({0} - {1})'.format(src, trg), file=sys.stderr)
            curr_dict.append(src_indices)
            curr_dict.append(trg_indices)
            train_pairs.append(curr_dict)
    if args.verbose:
        print('Normalizing embeddings...')
    # Step 0: Normalization
    for action in args.normalize:
        if action == 'unit':
            for i in range(len(emb)):
                emb[i] = embeddings.length_normalize(emb[i])
        elif action == 'center':
            for i in range(len(emb)):
                emb[i] = embeddings.mean_center(emb[i])
        elif action == 'unitdim':
            for i in range(len(emb)):
                emb[i] = embeddings.length_normalize_dimensionwise(emb[i])
        elif action == 'centeremb':
            for i in range(len(emb)):
                emb[i] = embeddings.mean_center_embeddingwise(emb[i])


    # Step 1: Optimization
    if args.verbose:
        print('Beginning Optimization')
    start_time = time.time()
    mean_size=0
    for tp in range(len(train_pairs)):
        src_indices = train_pairs[tp][2]
        trg_indices = train_pairs[tp][3]
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
        train_pairs[tp].append(uniq_src)
        train_pairs[tp].append(uniq_trg)
        train_pairs[tp].append(A)
        mean_size+= (len(uniq_src)*len(uniq_trg))
    mean_size = mean_size/len(train_pairs)
    np.random.seed(0)
    Lambda=args.l2_reg

    variables=[]
    manif = []
    low_rank=emb[0].shape[1]
    for i in range(len(emb)):
        variables.append(TT.matrix())
        manif.append(Stiefel(emb[i].shape[1],low_rank))
    variables.append(TT.matrix())
    manif.append(PositiveDefinite(low_rank))
    B = variables[-1]
    cost = 0.5*Lambda*(TT.sum(B**2))
    for i in range(len(train_pairs)):
        x = emb[train_pairs[i][0]]
        z = emb[train_pairs[i][1]]
        U1 = variables[train_pairs[i][0]]
        U2 = variables[train_pairs[i][1]]
        cost = cost + TT.sum(((shared(x[train_pairs[i][4]]).dot(U1.dot(B.dot(U2.T)))).dot(shared(z[train_pairs[i][5]]).T)-shared(train_pairs[i][6]))**2)/float(len(train_pairs[i][2]))
    solver = ConjugateGradient(maxtime=args.max_opt_time,maxiter=args.max_opt_iter,mingradnorm=1e-12)
    manifold =Product(manif)
    problem = Problem(manifold=manifold, cost=cost, arg=variables, verbosity=3)
    wopt = solver.solve(problem)
    w= wopt
    ### Save the models if requested
    if args.model_path is not None: 
        os.makedirs(args.model_path,exist_ok=True)

        for i in range(len(emb)):
            np.savetxt('{0}/U_{1}.csv'.format(args.model_path,lang_list[i]),wopt[i])

        np.savetxt('{}/B.csv'.format(args.model_path),wopt[-1])

        #with open('{}/lang_id_map.txt'.format(args.model_path),'w',encoding='utf-8') as idmapfile:
        #    for lang in lang_list:
        #        idmapfile.write(lang+'\n')


    # Step 2: Transformation
    Bhalf = scipy.linalg.sqrtm(wopt[-1])
    test_emb = []
    for i in range(len(emb)):
        test_emb.append(emb[i].dot(wopt[i]).dot(Bhalf))

    end_time = time.time()
    if args.verbose:
        print('Completed training in {0:.2f} seconds'.format(end_time-start_time))
    gc.collect()

    ### Save the GeoMM embeddings if requested
    if args.geomm_embeddings_path is not None: 
        os.makedirs(args.geomm_embeddings_path,exist_ok=True)
        for i in range(len(test_emb)):
            out_emb_fname=os.path.join(args.geomm_embeddings_path,'emb_{0}.vec'.format(lang_list[i]))
            with open(out_emb_fname,'w',encoding=args.encoding) as outfile:
                embeddings.write(words[i],embeddings.length_normalize(test_emb[i]),outfile)

    # Step 3: Evaluation
    if args.verbose:
        print('Beginning Evaluation')

    if args.normalize_eval:
        for i in range(len(test_emb)):
            test_emb[i] = embeddings.length_normalize(test_emb[i])

    # Loading test dictionary
    with open(args.dictionary_test_file, encoding=args.encoding, errors='surrogateescape') as ff:
        for line in ff:
            vals = line.split(',')
            curr_dict=[int(vals[0].strip()),int(vals[1].strip())]
            with open(vals[2].strip(), encoding=args.encoding, errors='surrogateescape') as f:
                src_word2ind = word2ind[curr_dict[0]]
                trg_word2ind = word2ind[curr_dict[1]]
                xw = test_emb[curr_dict[0]]
                zw = test_emb[curr_dict[1]]
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
                with cp.cuda.Device(1):
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
                print('Coverage:{0:7.2%}  Accuracy:{1:7.2%}  Accuracy(Top 5):{2:7.2%}  Accuracy(Top 10):{3:7.2%}'.format(coverage, accuracy, accuracy5, accuracy10))

if __name__ == '__main__':
    main()
