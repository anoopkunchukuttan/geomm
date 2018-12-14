# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#

mkdir -p vectors
curl -Lo vectors/wiki.en.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec
curl -Lo vectors/wiki.es.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.es.vec
curl -Lo vectors/wiki.fr.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.fr.vec
curl -Lo vectors/wiki.de.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.de.vec
curl -Lo vectors/wiki.ru.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.ru.vec
curl -Lo vectors/wiki.zh.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.zh.vec
curl -Lo vectors/wiki.it.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.it.vec
curl -Lo vectors/wiki.pt.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.pt.vec

aws_path='https://s3.amazonaws.com/arrival'
semeval_2017='http://alt.qcri.org/semeval2017/task2/data/uploads'

declare -A wordsim_lg
wordsim_lg=(["en"]="EN_MC-30.txt EN_MTurk-287.txt EN_RG-65.txt EN_VERB-143.txt EN_WS-353-REL.txt EN_YP-130.txt EN_MEN-TR-3k.txt EN_MTurk-771.txt EN_RW-STANFORD.txt EN_SIMLEX-999.txt EN_WS-353-ALL.txt EN_WS-353-SIM.txt" ["es"]="ES_MC-30.txt ES_RG-65.txt ES_WS-353.txt" ["de"]="DE_GUR350.txt DE_GUR65.txt DE_SIMLEX-999.txt DE_WS-353.txt DE_ZG222.txt" ["fr"]="FR_RG-65.txt" ["it"]="IT_SIMLEX-999.txt IT_WS-353.txt")

mkdir monolingual crosslingual

## Downloading en-{} or {}-en dictionaries
lgs="de en es fr ru zh"
mkdir -p crosslingual/dictionaries/
for lg in ${lgs}
do
  for suffix in .txt .0-5000.txt .5000-6500.txt
  do
    fname=en-$lg$suffix
    curl -Lo crosslingual/dictionaries/$fname $aws_path/dictionaries/$fname
    fname=$lg-en$suffix
    curl -Lo crosslingual/dictionaries/$fname $aws_path/dictionaries/$fname
  done
done

## Download European dictionaries
for src_lg in de es fr it pt
do
  for tgt_lg in de es fr it pt
  do
    if [ $src_lg != $tgt_lg ]
    then
      for suffix in .txt .0-5000.txt .5000-6500.txt
      do
        fname=$src_lg-$tgt_lg$suffix
        curl -Lo crosslingual/dictionaries/$fname $aws_path/dictionaries/european/$fname
      done
    fi
  done
done


## Monolingual wordsim tasks
for lang in "${!wordsim_lg[@]}"
do
  echo $lang
  mkdir monolingual/$lang
  for wsim in ${wordsim_lg[$lang]}
  do
    echo $wsim
    curl -Lo monolingual/$lang/$wsim $aws_path/$lang/$wsim
  done
done

## SemEval 2017 monolingual and cross-lingual wordsim tasks
# 1) Task1: monolingual
curl -Lo semeval2017-task2.zip $semeval_2017/semeval2017-task2.zip
unzip semeval2017-task2.zip

fdir='SemEval17-Task2/test/subtask1-monolingual'
for lang in en es de fa it
do
  mkdir -p monolingual/$lang
  uplang=`echo $lang | awk '{print toupper($0)}'`
  paste $fdir/data/$lang.test.data.txt $fdir/keys/$lang.test.gold.txt > monolingual/$lang/${uplang}_SEMEVAL17.txt
done

# 2) Task2: cross-lingual
mkdir -p crosslingual/wordsim
fdir='SemEval17-Task2/test/subtask2-crosslingual'
for lg_pair in de-es de-fa de-it en-de en-es en-fa en-it es-fa es-it it-fa
do
  echo $lg_pair
  paste $fdir/data/$lg_pair.test.data.txt $fdir/keys/$lg_pair.test.gold.txt > crosslingual/wordsim/$lg_pair-SEMEVAL17.txt
done
rm semeval2017-task2.zip
rm -r SemEval17-Task2/
