# 3DGAN
A 3d point cloud generative adversarial network, trained to produce digits 0-9.

The point cloud dataset was generated with Sidefx Houdini and the image below is a sample of generated digits.

# 
![plot](./3dgan/img/numbers.jpg)


Project is using Python 3.8.6 and Houdini 19.5.435

Run these commands to set up virtual environment for 3dgan on windows (git bash):
```
python -m venv ./venv/
source ./venv/Scripts/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install tqdm
```

Once you have the environment set up and all the files you can run training with:
```
python training.py
```
