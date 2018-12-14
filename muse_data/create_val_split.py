import sys
import random

lang1=str(sys.argv[1])
lang2=str(sys.argv[2])
f=open('muse_data/crosslingual/dictionaries/{0}-{1}.0-5000.txt'.format(lang1,lang2),'r',encoding='utf-8')
ff1=open('muse_data/crosslingual/dictionaries/{0}-{1}.train80.txt'.format(lang1,lang2),'w',encoding='utf-8')
ff2=open('muse_data/crosslingual/dictionaries/{0}-{1}.train20.txt'.format(lang1,lang2),'w',encoding='utf-8')
dict1=[]
for line in f:
	vals=line.split()
	dict1.append([vals[0].strip(),vals[1].strip()])

random.Random(int(sys.argv[3])).shuffle(dict1)
dict_size=len(dict1)
dict11=dict1[:int(0.8*dict_size)]
dict12=dict1[int(0.8*dict_size):]

for vals in dict11:
	ff1.write(vals[0].strip()+' '+vals[1].strip()+'\n')

for vals in dict12:
	ff2.write(vals[0].strip()+' '+vals[1].strip()+'\n')