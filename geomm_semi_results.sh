source activate testcross

VERBOSITY=0

# Table 7 Results
EMBEDDINGS="./muse_data/vectors"
DICTIONARIES="./muse_data/crosslingual/dictionaries"
echo "Table 7 Results"
echo "En-Es"
python "./muse_data/create_val_split.py" en es 42
python geomm_semi.py "$EMBEDDINGS/wiki.en.vec" "$EMBEDDINGS/wiki.es.vec" -dtrain "$DICTIONARIES/en-es.0-5000.txt" -dtest "$DICTIONARIES/en-es.5000-6500.txt" -dtrainspl "$DICTIONARIES/en-es.train80.txt" -dvalspl "$DICTIONARIES/en-es.train20.txt" --normalize unit center  --max_opt_iter 150 --l2_reg 1e-2 --max_vocab 200000 --normalize_eval --verbose $VERBOSITY
echo "Es-En"
python "./muse_data/create_val_split.py" es en 42
python geomm_semi.py "$EMBEDDINGS/wiki.es.vec" "$EMBEDDINGS/wiki.en.vec" -dtrain "$DICTIONARIES/es-en.0-5000.txt" -dtest "$DICTIONARIES/es-en.5000-6500.txt" -dtrainspl "$DICTIONARIES/es-en.train80.txt" -dvalspl "$DICTIONARIES/es-en.train20.txt" --normalize unit center  --max_opt_iter 150 --l2_reg 1e-2 --max_vocab 200000 --normalize_eval --verbose $VERBOSITY
echo "En-Fr"
python "./muse_data/create_val_split.py" en fr 42
python geomm_semi.py "$EMBEDDINGS/wiki.en.vec" "$EMBEDDINGS/wiki.fr.vec" -dtrain "$DICTIONARIES/en-fr.0-5000.txt" -dtest "$DICTIONARIES/en-fr.5000-6500.txt" -dtrainspl "$DICTIONARIES/en-fr.train80.txt" -dvalspl "$DICTIONARIES/en-fr.train20.txt" --normalize unit center  --max_opt_iter 150 --l2_reg 1e-1 --max_vocab 200000 --normalize_eval --verbose $VERBOSITY
echo "Fr-En"
python "./muse_data/create_val_split.py" fr en 42
python geomm_semi.py "$EMBEDDINGS/wiki.fr.vec" "$EMBEDDINGS/wiki.en.vec" -dtrain "$DICTIONARIES/fr-en.0-5000.txt" -dtest "$DICTIONARIES/fr-en.5000-6500.txt" -dtrainspl "$DICTIONARIES/fr-en.train80.txt" -dvalspl "$DICTIONARIES/fr-en.train20.txt" --normalize unit center  --max_opt_iter 150 --l2_reg 1e-1 --max_vocab 200000 --normalize_eval --verbose $VERBOSITY
echo "En-De"
python "./muse_data/create_val_split.py" en de 42
python geomm_semi.py "$EMBEDDINGS/wiki.en.vec" "$EMBEDDINGS/wiki.de.vec" -dtrain "$DICTIONARIES/en-de.0-5000.txt" -dtest "$DICTIONARIES/en-de.5000-6500.txt" -dtrainspl "$DICTIONARIES/en-de.train80.txt" -dvalspl "$DICTIONARIES/en-de.train20.txt" --normalize unit center  --max_opt_iter 150 --l2_reg 1e-1 --max_vocab 200000 --normalize_eval --verbose $VERBOSITY
echo "De-En"
python "./muse_data/create_val_split.py" de en 42
python geomm_semi.py "$EMBEDDINGS/wiki.de.vec" "$EMBEDDINGS/wiki.en.vec" -dtrain "$DICTIONARIES/de-en.0-5000.txt" -dtest "$DICTIONARIES/de-en.5000-6500.txt" -dtrainspl "$DICTIONARIES/de-en.train80.txt" -dvalspl "$DICTIONARIES/de-en.train20.txt" --normalize unit center  --max_opt_iter 150 --l2_reg 1e-2 --max_vocab 200000 --normalize_eval --verbose $VERBOSITY
echo "En-Ru"
python "./muse_data/create_val_split.py" en ru 42
python geomm_semi.py "$EMBEDDINGS/wiki.en.vec" "$EMBEDDINGS/wiki.ru.vec" -dtrain "$DICTIONARIES/en-ru.0-5000.txt" -dtest "$DICTIONARIES/en-ru.5000-6500.txt" -dtrainspl "$DICTIONARIES/en-ru.train80.txt" -dvalspl "$DICTIONARIES/en-ru.train20.txt" --normalize unit center  --max_opt_iter 150 --l2_reg 1e-1 --max_vocab 200000 --normalize_eval --verbose $VERBOSITY
echo "Ru-En"
python "./muse_data/create_val_split.py" ru en 42
python geomm_semi.py "$EMBEDDINGS/wiki.ru.vec" "$EMBEDDINGS/wiki.en.vec" -dtrain "$DICTIONARIES/ru-en.0-5000.txt" -dtest "$DICTIONARIES/ru-en.5000-6500.txt" -dtrainspl "$DICTIONARIES/ru-en.train80.txt" -dvalspl "$DICTIONARIES/ru-en.train20.txt" --normalize unit center  --max_opt_iter 150 --l2_reg 1e-1 --max_vocab 200000 --normalize_eval --verbose $VERBOSITY
echo "En-Zh"
python "./muse_data/create_val_split.py" en zh 5
python geomm_semi.py "$EMBEDDINGS/wiki.en.vec" "$EMBEDDINGS/wiki.zh.vec" -dtrain "$DICTIONARIES/en-zh.0-5000.txt" -dtest "$DICTIONARIES/en-zh.5000-6500.txt" -dtrainspl "$DICTIONARIES/en-zh.train80.txt" -dvalspl "$DICTIONARIES/en-zh.train20.txt" --normalize unit center  --max_opt_iter 150 --l2_reg 1e-2 --max_vocab 200000 --normalize_eval --verbose $VERBOSITY
echo "Zh-En"
python "./muse_data/create_val_split.py" zh en 5
python geomm_semi.py "$EMBEDDINGS/wiki.zh.vec" "$EMBEDDINGS/wiki.en.vec" -dtrain "$DICTIONARIES/zh-en.0-5000.txt" -dtest "$DICTIONARIES/zh-en.5000-6500.txt" -dtrainspl "$DICTIONARIES/zh-en.train80.txt" -dvalspl "$DICTIONARIES/zh-en.train20.txt" --normalize unit center  --max_opt_iter 150 --l2_reg 1e-2 --max_vocab 200000 --normalize_eval --verbose $VERBOSITY

EMBEDDINGS="./vecmap_data/embeddings"
DICTIONARIES="./vecmap_data/dictionaries"
echo "En-It"
python "./vecmap_data/create_val_split.py" en it 0
python geomm_semi.py "$EMBEDDINGS/en.emb.txt" "$EMBEDDINGS/it.emb.txt" -dtrain "$DICTIONARIES/en-it.train.txt" -dtest "$DICTIONARIES/en-it.test.txt" -dtrainspl "$DICTIONARIES/en-it.train80.txt" -dvalspl "$DICTIONARIES/en-it.train20.txt" --normalize unit center --max_opt_iter 150 --l2_reg 1e-1 --max_vocab 0 --normalize_eval --verbose $VERBOSITY
echo "It-En"
python "./vecmap_data/dictionary_reverse.py" en it
python "./vecmap_data/create_val_split.py" it en 0
python geomm_semi.py "$EMBEDDINGS/it.emb.txt" "$EMBEDDINGS/en.emb.txt" -dtrain "$DICTIONARIES/it-en.train.txt" -dtest "$DICTIONARIES/it-en.test.txt" -dtrainspl "$DICTIONARIES/it-en.train80.txt" -dvalspl "$DICTIONARIES/it-en.train20.txt" --normalize unit center --max_opt_iter 150 --l2_reg 1e-1 --max_vocab 0 --normalize_eval --verbose $VERBOSITY
echo "En-De"
python "./vecmap_data/create_val_split.py" en de 0
python geomm_semi.py "$EMBEDDINGS/en.emb.txt" "$EMBEDDINGS/de.emb.txt" -dtrain "$DICTIONARIES/en-de.train.txt" -dtest "$DICTIONARIES/en-de.test.txt" -dtrainspl "$DICTIONARIES/en-de.train80.txt" -dvalspl "$DICTIONARIES/en-de.train20.txt" --normalize unit center --max_opt_iter 150 --l2_reg 1e-1 --max_vocab 0 --normalize_eval --verbose $VERBOSITY
echo "En-Fi"
python "./vecmap_data/create_val_split.py" en fi 0
python geomm_semi.py "$EMBEDDINGS/en.emb.txt" "$EMBEDDINGS/fi.emb.txt" -dtrain "$DICTIONARIES/en-fi.train.txt" -dtest "$DICTIONARIES/en-fi.test.txt" -dtrainspl "$DICTIONARIES/en-fi.train80.txt" -dvalspl "$DICTIONARIES/en-fi.train20.txt" --normalize unit center --max_opt_iter 150 --l2_reg 1e-1 --max_vocab 0 --normalize_eval --verbose $VERBOSITY
echo "En-Es"
python "./vecmap_data/create_val_split.py" en es 0
python geomm_semi.py "$EMBEDDINGS/en.emb.txt" "$EMBEDDINGS/es.emb.txt" -dtrain "$DICTIONARIES/en-es.train.txt" -dtest "$DICTIONARIES/en-es.test.txt" -dtrainspl "$DICTIONARIES/en-es.train80.txt" -dvalspl "$DICTIONARIES/en-es.train20.txt" --normalize unit center --max_opt_iter 150 --l2_reg 1e-1 --max_vocab 0 --normalize_eval --verbose $VERBOSITY


source deactivate