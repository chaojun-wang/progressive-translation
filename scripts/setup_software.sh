path_to_top_dir= # path to the top directory (directory of readme)
codes_dir=$path_to_top_dir/codes
models_dir=$path_to_top_dir/models
data_dir=$path_to_top_dir/data
mkdir -p $codes_dir $models_dir $data_dir

cd $codes_dir

# download nematus
git clone https://github.com/EdinburghNLP/nematus.git

# download bpe-subword
git clone https://github.com/rsennrich/subword-nmt.git

# download moses
git clone https://github.com/moses-smt/mosesdecoder.git
cd mosesdecoder
# ref: http://www2.statmt.org/moses/?n=Development.GetStarted
./bjam -j4
cd ..

# download mgiza
# ref: https://github.com/transducens/mtl-da-emnlp#compile-mgiza
git clone https://github.com/moses-smt/mgiza.git
cd mgiza/mgizapp
mkdir build && cd build
cmake ..
make
ln -s $PWD/../scripts/merge_alignment.py $PWD/bin/merge_alignment.py
