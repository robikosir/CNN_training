# CNN training and evaulation
This project is used to recognize ethnicity and gender from EAR.
## Dataset
In order to run training model, you have to download following [Dataset.](http://awe.fri.uni-lj.si/datasets.html). In case you don't have premission to do so, feel free to use any other biometric dataset intended for recognition dataset that has annotations that you want to train on.
## Install required requirements
First run `pip install -r requirements.txt`
## CNN Training
- Run `python train.py`

This saves your model to ˙mode.h5`. The model is now ready to use in our prediciton python script.

##Predict
- Load your images or choose the one that is already uploaded to the root directory (`test.png`)
- Run `python predict.py`

This saves the image in `/results` with the corresponding gender and ethnicity information

## Results
Running at `epochs=100`:
- gender accuracy: 92%
- ethnicity accuracy: 84%

## Example outputs
![alt text](https://github.com/123robi/CNN_training/blob/main/results/man_asian.png?raw=true)
![alt text](https://github.com/123robi/CNN_training/blob/main/results/south_american.png?raw=true)
