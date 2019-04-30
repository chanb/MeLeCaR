# Supervised Learning Approach Part

We attempt to create a cache eviction policy using multi-label multi-class supervised learning and reinforcement learning. For reinforcement learning context, we treat the problem as a contextual bandit problem, with action space equals to the cache size. Below is our supervised learning part of code. 

We formulate this problem as a multi-label binary-class classification. Due to the nature of the problem, we decide to use GRU model with sequence length of 5.

### **Environment**

We have used `casa-09`, `casa-12`, `cheetah-12` and `cheetah-16` as our training or testing (evaluating) workload. If you want to run the code, just put them into the root directory.

### **Data Generation**

We use the ipython notebook "Generate Data.ipynb" to generate data. Put ".blkparse" file into root derectory, and after run the ipython notebook, you will get ".pkl" files. We will use these ".pkl" files to train or test our model.

### **Models**

The models are stored in `models`. We currently use GRU as our default architecture choice.

### **Evaluation**

We use the ipython notebook "Evaluation.ipynb" to evaluate our OPT, LFU and LRU cache replacement policies. We use "predict.py" to evaluate our trained models.

### **Visualization**

We use the ipython notebook "Chart for Supervised Learning.ipynb" to plot our experiment results. Results are stored in the "result.xlsx" excel file.


### **Scripts**

There are two scripts that are relevant to training and evaluating a supervised learning model: `train.py` and `predict.py`.

- `train.py`: This script trains a supervised learning model given a specific workload and cache size.  

  Example training: `python train.py --train_data=online_12.pkl --output_dir=train_output --batch_size=100 --learning_rate=0.0005 --num_epochs=2000 --model_type=gru --model_name=online_12_train_output`

- `predict.py`: This script uses trained model to predict specific workload.

  Example evaluation: `python predict.py --train_data=online_09.pkl --trained_model_file_name=./train_output/online_09_train_output-gru.h5 --predict_pkl_data=online_12.pkl --predict_raw_file=online.cs.fiu.edu-110108-113008.12.blkparse --data_percentage=0.1 --save_result_file=train_online09_C30_predict_online12 --batch_size=100 --cache_size=100`
