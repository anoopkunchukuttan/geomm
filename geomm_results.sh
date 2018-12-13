source activate testcross

# Table 1 Results
EMBEDDINGS="./muse_data/vectors"
DICTIONARIES="./muse_data/crosslingual/dictionaries"
echo "Table 1 Results"
echo "En-Es"
python geomm.py "$EMBEDDINGS/wiki.en.vec" "$EMBEDDINGS/wiki.es.vec" -dtrain "$DICTIONARIES/en-es.0-5000.txt" -dtest "$DICTIONARIES/en-es.5000-6500.txt" --normalize unit center  --max_opt_iter 150 --l2_reg 1e3 --max_vocab 200000 --normalize_eval
echo "Es-En"
python geomm.py "$EMBEDDINGS/wiki.es.vec" "$EMBEDDINGS/wiki.en.vec" -dtrain "$DICTIONARIES/es-en.0-5000.txt" -dtest "$DICTIONARIES/es-en.5000-6500.txt" --normalize unit center  --max_opt_iter 150 --l2_reg 1e2 --max_vocab 200000 --normalize_eval
echo "En-Fr"
python geomm.py "$EMBEDDINGS/wiki.en.vec" "$EMBEDDINGS/wiki.fr.vec" -dtrain "$DICTIONARIES/en-fr.0-5000.txt" -dtest "$DICTIONARIES/en-fr.5000-6500.txt" --normalize unit center  --max_opt_iter 150 --l2_reg 1e2 --max_vocab 200000 --normalize_eval
echo "Fr-En"
python geomm.py "$EMBEDDINGS/wiki.fr.vec" "$EMBEDDINGS/wiki.en.vec" -dtrain "$DICTIONARIES/fr-en.0-5000.txt" -dtest "$DICTIONARIES/fr-en.5000-6500.txt" --normalize unit center  --max_opt_iter 150 --l2_reg 1e3 --max_vocab 200000 --normalize_eval
echo "En-De"
python geomm.py "$EMBEDDINGS/wiki.en.vec" "$EMBEDDINGS/wiki.de.vec" -dtrain "$DICTIONARIES/en-de.0-5000.txt" -dtest "$DICTIONARIES/en-de.5000-6500.txt" --normalize unit center  --max_opt_iter 150 --l2_reg 1e3 --max_vocab 200000 --normalize_eval
echo "De-En"
python geomm.py "$EMBEDDINGS/wiki.de.vec" "$EMBEDDINGS/wiki.en.vec" -dtrain "$DICTIONARIES/de-en.0-5000.txt" -dtest "$DICTIONARIES/de-en.5000-6500.txt" --normalize unit center  --max_opt_iter 150 --l2_reg 1e2 --max_vocab 200000 --normalize_eval
echo "En-Ru"
python geomm.py "$EMBEDDINGS/wiki.en.vec" "$EMBEDDINGS/wiki.ru.vec" -dtrain "$DICTIONARIES/en-ru.0-5000.txt" -dtest "$DICTIONARIES/en-ru.5000-6500.txt" --normalize unit center  --max_opt_iter 150 --l2_reg 1e3 --max_vocab 200000 --normalize_eval
echo "Ru-En"
python geomm.py "$EMBEDDINGS/wiki.ru.vec" "$EMBEDDINGS/wiki.en.vec" -dtrain "$DICTIONARIES/ru-en.0-5000.txt" -dtest "$DICTIONARIES/ru-en.5000-6500.txt" --normalize unit center  --max_opt_iter 150 --l2_reg 1e2 --max_vocab 200000 --normalize_eval
echo "En-Zh"
python geomm.py "$EMBEDDINGS/wiki.en.vec" "$EMBEDDINGS/wiki.zh.vec" -dtrain "$DICTIONARIES/en-zh.0-5000.txt" -dtest "$DICTIONARIES/en-zh.5000-6500.txt" --normalize unit center  --max_opt_iter 150 --l2_reg 1e2 --max_vocab 200000 --normalize_eval
echo "Zh-En"
python geomm.py "$EMBEDDINGS/wiki.zh.vec" "$EMBEDDINGS/wiki.en.vec" -dtrain "$DICTIONARIES/zh-en.0-5000.txt" -dtest "$DICTIONARIES/zh-en.5000-6500.txt" --normalize unit center  --max_opt_iter 150 --l2_reg 1e2 --max_vocab 200000 --normalize_eval

# Table 2 Results
EMBEDDINGS="./vecmap_data/embeddings"
DICTIONARIES="./vecmap_data/dictionaries"
echo "Table 2 Results"
languages=( it de fi es )
for lang in "${languages[@]}"
do
	echo "en-$lang"
	python geomm.py "$EMBEDDINGS/en.emb.txt" "$EMBEDDINGS/$lang.emb.txt" -dtrain "$DICTIONARIES/en-$lang.train.txt" -dtest "$DICTIONARIES/en-$lang.test.txt" --normalize unit center --max_opt_iter 150 --l2_reg 1e3 --max_vocab 0 --normalize_eval
done

# Table 6 Results
EMBEDDINGS="./muse_data/vectors"
DICTIONARIES="./muse_data/crosslingual/dictionaries"
echo "Table 6 Results"
echo "Fr-It-Pt"
python geomm_cmp_pip.py "$EMBEDDINGS/wiki.fr.vec" "$EMBEDDINGS/wiki.it.vec" "$EMBEDDINGS/wiki.pt.vec" -dtrain1 "$DICTIONARIES/fr-it.excfull.txt" -dtrain2 "$DICTIONARIES/it-pt.excfull.txt" -dtest "$DICTIONARIES/fr-pt.5000-6500.txt"  --normalize unit center  --max_opt_iter 150 --l2_reg 1e2 --max_vocab 200000 --normalize_eval
echo "It-De-Es"
python geomm_cmp_pip.py "$EMBEDDINGS/wiki.it.vec" "$EMBEDDINGS/wiki.de.vec" "$EMBEDDINGS/wiki.es.vec" -dtrain1 "$DICTIONARIES/it-de.excfull.txt" -dtrain2 "$DICTIONARIES/de-es.excfull.txt" -dtest "$DICTIONARIES/it-es.5000-6500.txt"  --normalize unit center  --max_opt_iter 150 --l2_reg 1e2 --max_vocab 200000 --normalize_eval
echo "Es-Pt-Fr"
python geomm_cmp_pip.py "$EMBEDDINGS/wiki.es.vec" "$EMBEDDINGS/wiki.pt.vec" "$EMBEDDINGS/wiki.fr.vec" -dtrain1 "$DICTIONARIES/es-pt.excfull.txt" -dtrain2 "$DICTIONARIES/pt-fr.excfull.txt" -dtest "$DICTIONARIES/es-fr.5000-6500.txt"  --normalize unit center  --max_opt_iter 150 --l2_reg 1e2 --max_vocab 200000 --normalize_eval

source deactivate