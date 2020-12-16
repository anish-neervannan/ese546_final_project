## The Effect of Genetic Algorithms and Attention Modules on Long Short Term Memory Networks for Stock Forecasting


### ESE 546 Final Project


### Authors
- Anish Neervannan (anishrn@seas.upenn.edu)
- Vinay Senthil (vinayksk@seas.upenn.edu)
- Bhaskar Abhiraman (bhaskara@seas.upenn.edu)


### Abstract
In this report, we characterize the performance of long short term memory (LSTM) networks for stock forecasting and study how they are affected by attention modules and genetic algorithms. We find that due to the highly stochastic nature of stock prices, LSTM predictions are comparable to random guessing. Attention modules marginally improve this performance. Because genetic algorithms for updating weights ignore the gradient, they require significant training times in order to achieve useful changes in loss; nevertheless, due to the challenging domain of stock prediction, genetic algorithms could be a useful tool in tandem with traditional gradient descent methods.  


### Overview of Code
Our code is organized into directories and scripts. We have the following directories:

- **data:** contains the raw Kaggle data, preprocessed and stored into .npy files separated by sector and training/validation vs test set
- **notebooks:** contains the raw draft ipynb notebooks that were used initially to download the data and preprocess it. The data was stored locally after the initial download, and is now uploaded to the data/ folder, so no need to use these anymore
- **results:** after training and evaluation, the scripts store plots of the training/validation loss, the trained model weights, and the raw npy files used to generate the graphs

Additionally, we have the following scripts:

- **eval.py:** this will load the models saved into the results/ directory and run the test set through the model to compute directional accuracy and MAE Loss
- **genetic_algorithm.py:** contains the methods needed to randomly initialize models, mutate, and spawn more networks
- **nn_architectures.py:** contains the modules for the base LSTM and the attention LSTM
- **process_data.py:** this is a Python version of the .ipynb notebook stored in the notebooks/ directory that was used to generate and save the training/validation and test sets
- **train.py** the main script that takes in a flag for the model to train (see (Instructions)[#instructions] below). Loads in the required dataset, scales it, initializes a model, trains it for a pre-defined set of hyperparameters including epochs, batches, sequence length, generations, number of models to spawn, etc. After each epoch/generation, it computes validation loss, and at the end of training saves the plots and .npy files containing the losses themselves.


### Instructions
1. Clone this git repository
2. Run the following command to train the network: `python train.py {model name}` where model name is: `base`, `attention`, `base_ga`, or `attention_ga`. The default dataset that is used is "Information Technology," but this can be changed on line 39 of train.py. To see a full list of all available sectors, refer to data/ folder.
3. After running all four models, run the following to evaluate the network performances: `python eval.py`. All of the graphs will be saved to the results/ folder and the test error/directional accuracies should print to the terminal screen
