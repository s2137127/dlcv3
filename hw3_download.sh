#!/bin/bash
cd hw3_1
python -c "import clip; clip.load('ViT-B/32')"
cd ../hw3_2
wget -O checkpoint_25.pth https://www.dropbox.com/s/n4zngffhrc1ov85/checkpoint_25.pth?dl=1
