ln -s /data ~/hpdic/data
cd ~/hpdic/data
wget -qO file_list.txt https://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/file_list.txt

CHUNK_URL=$(grep "chunk_0000.bvecs" file_list.txt)

echo "Target URL: $CHUNK_URL"

sudo apt install -y axel wget

if command -v axel &> /dev/null; then
    axel -n 16 -a -o chunk_0000.bvecs "$CHUNK_URL"
else
    wget -c -O chunk_0000.bvecs "$CHUNK_URL"
fi