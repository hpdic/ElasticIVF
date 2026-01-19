# HPDIC MOD of FAISS
We assume you are using (e.g., Chameleon Cloud `nc33` at U. Chicago) Ubuntu 24.04, NVIDIA RTX 6000 GPU (24 GB RAM, Driver 560.35.05, CUDA 12.6), 192 GB RAM, Intel(R) Xeon(R) Gold 6126 CPU @ 2.60GHz (48 Cores).

## Recompile C++
```bash
# Recompile Faiss with HPDIC modifications:
cd ~/ElasticIVF
cmake -B build . \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=ON \
    -DBUILD_TESTING=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="75" \
    -DPython_EXECUTABLE=$(which python)
make -C build -j faiss

# SIVF add benchmark:
cd ~/ElasticIVF/build
make -j test_sivf_add
./faiss/gpu/test_sivf_add

# SIVF search benchmark:
cd ~/ElasticIVF/build
make -j test_sivf_search
./faiss/gpu/test_sivf_search

# SIVF delete benchmark:
cd ~/ElasticIVF/build
make -j test_sivf_delete
./faiss/gpu/test_sivf_delete

# SIVF sensitivity benchmark:
cd ~/ElasticIVF/build
make -j test_sivf_sensitivity
./faiss/gpu/test_sivf_sensitivity

# Dataset Deep1B:
cd ~/ElasticIVF/build
make test_sivf_deep_add test_sivf_deep_search test_sivf_deep_delete -j
./test_sivf_deep_add
./test_sivf_deep_search
./test_sivf_deep_delete

# Dataset SIFT1M:
cd ~/ElasticIVF/build
make -j test_sivf_sift_add
./test_sivf_sift_add
make -j test_sivf_sift_search
./test_sivf_sift_search
make -j test_sivf_sift_delete
./test_sivf_sift_delete

# Dataset GIST1M:
cd ~/ElasticIVF/build
make test_sivf_gist_add test_sivf_gist_search test_sivf_gist_delete -j
./test_sivf_gist_add
./test_sivf_gist_search
./test_sivf_gist_delete

# Dataset T2I-1B:
cd ~/ElasticIVF/build
make test_sivf_t2i_add test_sivf_t2i_search test_sivf_t2i_delete -j
./test_sivf_t2i_add
./test_sivf_t2i_search
./test_sivf_t2i_delete

# Sliding window:
cd ~/ElasticIVF/build
make -j test_sivf_sliding
./test_sivf_sliding # default is SIFT
./test_sivf_sliding gist

# Memory usage:
cd ~/ElasticIVF/build
make -j test_sivf_memory
./test_sivf_memory # default is SIFT
./test_sivf_memory gist

# SIVF vs. Non-IVF:
cd ~/ElasticIVF/build
make -j test_sivf_nonivf
./test_sivf_nonivf
```

## Benchmarks
```bash
# Download SIFT1M dataset:
bash ~/ElasticIVF/hpdic/script/download_sift.sh

# Quick Python test with precompiled Faiss library:
source ~/ElasticIVF/myenv/bin/activate
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libmkl_rt.so python3 ~/ElasticIVF/hpdic/script/benchmark_baseline.py

# Serious C++ test with local compilation of Faiss source code:
cp ~/ElasticIVF/hpdic/config/c_cpp_properties.json ~/.vscode/.
cd ~/ElasticIVF/hpdic/experiment
g++ -O3 -std=c++17 -fopenmp benchmark_baseline.cpp -o benchmark_baseline.bin \
    -I/home/cc/ElasticIVF \
    -I/usr/local/cuda/include \
    -L/home/cc/ElasticIVF/build/faiss \
    -L/usr/local/cuda/lib64 \
    -lfaiss \
    -lopenblas \
    -lcudart \
    -lcublas
./benchmark_baseline.bin
```

## Installation
```bash
git config --global user.name "Dongfang Zhao"
git config --global user.email "dzhao@uw.edu"
sudo apt install -y cmake ccache swig g++ libopenblas-dev libmkl-dev git
cd ~
git clone https://github.com/hpdic/ElasticIVF.git
cd ElasticIVF
python3 -m venv myenv
source myenv/bin/activate
which python # e.g., /home/cc/ElasticIVF/myenv/bin/python
pip install matplotlib torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python3 ~/ElasticIVF/hpdic/script/test_gpu.py
rm -rf build
cmake -B build . \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=ON \
    -DFAISS_ENABLE_RAFT=OFF \
    -DBUILD_TESTING=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="75" \
    -DPython_EXECUTABLE=$(which python)
make -C build -j $(nproc)
cd build/faiss/python
python setup.py install
```

## Test
```bash
cd ~/ElasticIVF/tutorial/cpp
# For Intel MKL
g++ 4-GPU.cpp -o 4-GPU.bin \
    -fopenmp \
    -I ../.. \
    -I /usr/local/cuda/include \
    -L ../../build/faiss -lfaiss \
    -L /usr/local/cuda/lib64 -lcudart -lcublas \
    -lmkl_rt \
    -Wl,-rpath=$(pwd)/../../build/faiss
./4-GPU.bin

# For OpenBLAS
g++ 4-GPU.cpp -o 4-GPU.bin \
    -fopenmp \
    -I ../.. \
    -I /usr/local/cuda/include \
    -L ../../build/faiss -lfaiss \
    -L /usr/local/cuda/lib64 -lcudart -lcublas \
    -lopenblas \
    -Wl,-rpath=$(pwd)/../../build/faiss
./4-GPU.bin
```

## Reinstall Python package
```bash
cd ~/ElasticIVF/build/faiss/python
python setup.py install
```
