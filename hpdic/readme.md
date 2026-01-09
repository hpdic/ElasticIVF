# HPDIC MOD of FAISS
We assume you are using (e.g., Chameleon Cloud `nc33` at U. Chicago) Ubuntu 24.04, NVIDIA RTX 6000 GPU (24 GB RAM, Driver 560.35.05, CUDA 12.6), 192 GB RAM, Intel(R) Xeon(R) Gold 6126 CPU @ 2.60GHz (48 Cores).

## Installation
```bash
git config --global user.name "Dongfang Zhao"
git config --global user.email "dzhao@uw.edu"
sudo apt install -y cmake swig g++ libopenblas-dev libmkl-dev git
cd ~
git clone https://github.com/hpdic/ElasticIVF.git
cd ElasticIVF
python3 -m venv myenv
source myenv/bin/activate
which python 
# 输出应该是 /home/cc/ElasticIVF/myenv/bin/python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python3 ~/ElasticIVF/hpdic/script/test_gpu.py
rm -rf build
cmake -B build . \
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

## Recompile C++
```bash
cd ~/ElasticIVF
make -C build -j $(nproc) faiss
```

## Reinstall Python package
```bash
cd ~/ElasticIVF/build/faiss/python
python setup.py install
```
