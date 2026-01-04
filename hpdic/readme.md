# HPDIC MOD of FAISS
We assume you are using (e.g., Chameleon Cloud `gigaio05`) Ubuntu 24.04, NVIDIA A100 GPU (40 GB RAM, Driver 560.35.05, CUDA 12.6), 256 GB RAM, Intel(R) Xeon(R) Platinum 8380 CPU (160 Cores).

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
# For OpenBLAS
g++ 4-GPU.cpp -o 4-GPU.bin \
    -fopenmp \
    -I ../.. \
    -I /usr/local/cuda/include \
    -L ../../build/faiss -lfaiss \
    -L /usr/local/cuda/lib64 -lcudart -lcublas \
    -lopenblas \
    -Wl,-rpath=$(pwd)/../../build/faiss
```

## Installation
```bash
git config --global user.name "Dongfang Zhao"
git config --global user.email "dzhao@uw.edu"
python3 script/test_gpu.py
python3 -m venv myenv
source myenv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
sudo apt install -y cmake swig g++ libopenblas-dev libmkl-dev git
cd ~
git clone https://github.com/hpdic/ElasticIVF.git
cd ElasticIVF
source ~/myenv/bin/activate
which python 
# 输出应该是 /home/cc/myenv/bin/python
rm -rf build
cmake -B build . \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=ON \
    -DFAISS_ENABLE_RAFT=OFF \
    -DBUILD_TESTING=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="80" \
    -DPython_EXECUTABLE=$(which python)
make -C build -j $(nproc)
cd build/faiss/python
python setup.py install
```