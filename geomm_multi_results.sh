source activate testcross

create_emb_file() {
	touch $1
	if [ "$2" = "muse" ]
	then
		echo "$EMBEDDINGS/wiki.en.vec" >> $1
		echo "$EMBEDDINGS/wiki.es.vec" >> $1
		echo "$EMBEDDINGS/wiki.fr.vec" >> $1
		echo "$EMBEDDINGS/wiki.de.vec" >> $1
		echo "$EMBEDDINGS/wiki.ru.vec" >> $1
		echo "$EMBEDDINGS/wiki.zh.vec" >> $1
	else
		echo "$EMBEDDINGS/en.emb.txt" >> $1
		echo "$EMBEDDINGS/it.emb.txt" >> $1
		echo "$EMBEDDINGS/de.emb.txt" >> $1
		echo "$EMBEDDINGS/fi.emb.txt" >> $1
		echo "$EMBEDDINGS/es.emb.txt" >> $1
	fi
}

create_emb_tri() {
	touch $1	
	echo "$EMBEDDINGS/wiki.$2.vec" >> $1
	echo "$EMBEDDINGS/wiki.$3.vec" >> $1
	echo "$EMBEDDINGS/wiki.$4.vec" >> $1
}

create_dict_file() {
	touch $1
	if [ "$2" = "muse" ]
	then
		if [ "$3" = "l-en" ]
		then
			echo "1, 0, $DICTIONARIES/es-en.$4.txt" >> $1
			echo "2, 0, $DICTIONARIES/fr-en.$4.txt" >> $1
			echo "3, 0, $DICTIONARIES/de-en.$4.txt" >> $1
			echo "4, 0, $DICTIONARIES/ru-en.$4.txt" >> $1
			echo "5, 0, $DICTIONARIES/zh-en.$4.txt" >> $1
		else
			echo "0, 1, $DICTIONARIES/en-es.$4.txt" >> $1
			echo "0, 2, $DICTIONARIES/en-fr.$4.txt" >> $1
			echo "0, 3, $DICTIONARIES/en-de.$4.txt" >> $1
			echo "0, 4, $DICTIONARIES/en-ru.$4.txt" >> $1
			echo "0, 5, $DICTIONARIES/en-zh.$4.txt" >> $1
		fi
	else
		echo "0, 1, $DICTIONARIES/en-it.$4.txt" >> $1
		echo "0, 2, $DICTIONARIES/en-de.$4.txt" >> $1
		echo "0, 3, $DICTIONARIES/en-fi.$4.txt" >> $1
		echo "0, 4, $DICTIONARIES/en-es.$4.txt" >> $1
	fi
}

create_dict_tri() {
	touch $1
	if [ "$5" = "train" ]
	then
		echo "0, 1, $DICTIONARIES/$2-$3.excfull.txt" >> $1
		echo "1, 2, $DICTIONARIES/$3-$4.excfull.txt" >> $1
	else
		echo "0, 2, $DICTIONARIES/$2-$4.5000-6500.txt" >> $1
	fi
}


# Table 1
EMBEDDINGS="./muse_data/vectors"
DICTIONARIES="./muse_data/crosslingual/dictionaries"
echo "Table 1 Results"
echo "En-Es, En-Fr, En-De, En-Ru, En-Zh"
[ -e emb-muse-en-l.txt ] && rm emb-muse-en-l.txt; [ -e train-muse-en-l.txt ] && rm train-muse-en-l.txt; [ -e test-muse-en-l.txt ] && rm test-muse-en-l.txt;
create_emb_file emb-muse-en-l.txt muse; create_dict_file train-muse-en-l.txt muse en-l 0-5000; create_dict_file test-muse-en-l.txt muse en-l 5000-6500;
python geomm_multi.py emb-muse-en-l.txt -dtrain_file train-muse-en-l.txt -dtest_file test-muse-en-l.txt --normalize unit center --l2_reg 1e-1 --max_opt_iter 600 --normalize_eval --max_vocab 200000
rm emb-muse-en-l.txt; rm train-muse-en-l.txt; rm test-muse-en-l.txt;
echo "Es-En, Fr-En, En-De-En, Ru-En, Zh-En"
[ -e emb-muse-l-en.txt ] && rm emb-muse-l-en.txt; [ -e train-muse-l-en.txt ] && rm train-muse-l-en.txt; [ -e test-muse-l-en.txt ] && rm test-muse-l-en.txt;
create_emb_file emb-muse-l-en.txt muse; create_dict_file train-muse-l-en.txt muse l-en 0-5000; create_dict_file test-muse-l-en.txt muse l-en 5000-6500;
python geomm_multi.py emb-muse-l-en.txt -dtrain_file train-muse-l-en.txt -dtest_file test-muse-l-en.txt --normalize unit center --l2_reg 1e-1 --max_opt_iter 600 --normalize_eval --max_vocab 200000
rm emb-muse-l-en.txt; rm train-muse-l-en.txt; rm test-muse-l-en.txt;


# Table 2
EMBEDDINGS="./vecmap_data/embeddings"
DICTIONARIES="./vecmap_data/dictionaries"
echo "Table 2 Results"
echo "En-It, En-De, En-Fi, En-Es"
[ -e emb-vecmap.txt ] && rm emb-vecmap.txt; [ -e train-vecmap.txt ] && rm train-vecmap.txt; [ -e test-vecmap.txt ] && rm test-vecmap.txt;
create_emb_file emb-vecmap.txt vecmap; create_dict_file train-vecmap.txt vecmap en-l train; create_dict_file test-vecmap.txt vecmap en-l test;
python geomm_multi.py emb-vecmap.txt -dtrain_file train-vecmap.txt -dtest_file test-vecmap.txt --normalize unit center --l2_reg 1e0 --max_opt_iter 600 --normalize_eval --max_vocab 0
rm emb-vecmap.txt; rm train-vecmap.txt; rm test-vecmap.txt;


# Table 6
EMBEDDINGS="./muse_data/vectors"
DICTIONARIES="./muse_data/crosslingual/dictionaries"
echo "Table 6 Results"

echo "Fr-It-Pt"
[ -e emb-tri.txt ] && rm emb-tri.txt; [ -e train-tri.txt ] && rm train-tri.txt; [ -e test-tri.txt ] && rm test-tri.txt;
create_emb_tri emb-tri.txt fr it pt; create_dict_tri train-tri.txt fr it pt train; create_dict_tri test-tri.txt fr it pt test;
python geomm_multi.py emb-tri.txt -dtrain_file train-tri.txt -dtest_file test-tri.txt --normalize unit center --l2_reg 1e0 --max_opt_iter 200 --normalize_eval --max_vocab 200000
rm emb-tri.txt; rm train-tri.txt; rm test-tri.txt;

echo "It-De-Es"
[ -e emb-tri.txt ] && rm emb-tri.txt; [ -e train-tri.txt ] && rm train-tri.txt; [ -e test-tri.txt ] && rm test-tri.txt;
create_emb_tri emb-tri.txt it de es; create_dict_tri train-tri.txt it de es train; create_dict_tri test-tri.txt it de es test;
python geomm_multi.py emb-tri.txt -dtrain_file train-tri.txt -dtest_file test-tri.txt --normalize unit center --l2_reg 1e0 --max_opt_iter 200 --normalize_eval --max_vocab 200000
rm emb-tri.txt; rm train-tri.txt; rm test-tri.txt;

echo "Es-Pt-Fr"
[ -e emb-tri.txt ] && rm emb-tri.txt; [ -e train-tri.txt ] && rm train-tri.txt; [ -e test-tri.txt ] && rm test-tri.txt;
create_emb_tri emb-tri.txt es pt fr; create_dict_tri train-tri.txt es pt fr train; create_dict_tri test-tri.txt es pt fr test;
python geomm_multi.py emb-tri.txt -dtrain_file train-tri.txt -dtest_file test-tri.txt --normalize unit center --l2_reg 1e-1 --max_opt_iter 200 --normalize_eval --max_vocab 200000
rm emb-tri.txt; rm train-tri.txt; rm test-tri.txt;




source deactivate testcross