# Grab challenge: Computer Vision solution
## Getting started
Notice that I have prepared the requirements.txt file
```
1. Make a env (python3.6)
2. Install requirements.txt
```
## Training
```
1. mkdir data-processed
2. cd data
3. wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
4. wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz
5. tar zxvf cars_train.tgz
6. tar zxvf cars_test.tgz
7. cd ..
8. python3 preprocess.py
9. python3 train.py
```
### Training baseline 
![](https://github.com/thaiduongx26/grab_submit/blob/master/image/baseline-training.png)

## Testing
Download [model_grab_challenge.zip](https://drive.google.com/file/d/1jj-dv_Pe_w2nvLrMsTqk0SioWRfPj1zG/view?usp=sharing)
```
1. Unzip model_grab_challenge.zip to models/
2. 2 options to run test: 
    python3 predict.py --image /path/to/image
    python3 predict.py --testdir /path/to/dir
```
The option --image will show result in terminal </br>
The option --testdir will save result in csv file (data.csv)
### Testing baseline
![alt text](https://github.com/thaiduongx26/grab_submit/blob/master/image/baseline-testing.png)
