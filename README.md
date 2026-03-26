The code for the final project assignment is the notebook FakeNewsFinalProject.ipynb.

The notebook FakeNewsFinalProject.ipynb creates a fake news prediction system. The notebook contains the entire code to answer the assignment, and covers the full Data Science pipeline, from data processing, to modelling, to visualization and interpretation. The notebook evaluates on both the full FakeNewsCorpus and a cross-domain, the LIAR dataset.

The python program uses version: 3.8 or higher, and requires the following modules:
- numpy  
- pandas  
- matplotlib  
- scikit-learn  
- nltk
- re
- time
- collections
- scipy

The notebook is divided into multiple parts corresponding to the different parts of the assignment.
Part 1 will overall do the following:
1. Load and clean a sample of the FakeNewsCorpus dataset (fakenews_sample.csv)
2. Preprocess the dataset, by tokenizating, removing stopword, and stemming
3. Compute vocabulary sizes and reduction rates after preprocessing
4. Apply the same preprocessing pipeline to the full FakeNewsCorpus, using chunking
5. Explore different statistcs of the full FakeNewsCorpus and create plots
6. Split the dataset into training, validation, and test sets (80/10/10)

Part 2 will overall do the following:
1. Train and validate a logistic regression model
3. Train and validate a logistic model on metadata features (title, content and domain)

Part 3 will overall do the following:
1. Create an adavanced model: a Neural Network
2. Train and validate the NN with TF-IDF features

Part 4 will overall do the following:
1. Evaluate both models on the FakeNewsCorpus test set
2. Evaluate the same models on the LIAR dataset (test.tsv)
4. Print the results using confusion matrices

Part 5 will overall do the following:
1. Create plots of the confusion matrices
2. Plot the accuracies of the evaluations
3. Plot the F1-scores of the evaluations

To run the programs, press the "Run All" button in the upper bar of the notebook. For the code to run, the listed files must be in the same directory of the notebook. The generated images will automatically show in the notebooks. Some files will be saved in the directory of the notebook.