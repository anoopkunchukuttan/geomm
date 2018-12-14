import sys
import os

lang1=str(sys.argv[1])
lang2=str(sys.argv[2])
f=open(os.path.join("vecmap_data",os.path.join("dictionaries",lang1+'-'+lang2+'.train.txt')),'r',encoding='utf-8')
ff=open(os.path.join("vecmap_data",os.path.join("dictionaries",lang2+'-'+lang1+'.train.txt')),'w',encoding='utf-8')
for line in f:
	vals=line.split()
	ff.write(vals[1].strip()+' '+vals[0].strip()+'\n')

f=open(os.path.join("vecmap_data",os.path.join("dictionaries",lang1+'-'+lang2+'.test.txt')),'r',encoding='utf-8')
ff=open(os.path.join("vecmap_data",os.path.join("dictionaries",lang2+'-'+lang1+'.test.txt')),'w',encoding='utf-8')
for line in f:
	vals=line.split()
	ff.write(vals[1].strip()+' '+vals[0].strip()+'\n')