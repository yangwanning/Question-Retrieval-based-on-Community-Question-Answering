# Question-Retrieval-based-on-Community-Question-Answering


This is my thesis project that amis at using question retrieval techniques to faciliate Community Question Answering (CQA) platforms.

This repo includes following part:</b>

1. **Datasets**  
Two datasets I used for my thesis are ***Quora*** and ***SemEval***. ***Quora*** was built from [Quora Duplicated Question Pairs](https://www.kaggle.com/c/quora-question-pairs "悬停显示") and ***SemEval*** was directly obtained from a commonly-used benchmark dataset: [SemEval-Task3](http://alt.qcri.org/semeval2017/task3/). Each data sample consists of one original question (new question asked by users) and a certain number of candidate questions (historical questions).

2. **Models**  
a) <b>Traditional Models</b>  
I firstly decided to try two typical traditional models (<b>BM25</b> and Query Likelihood (QL) model) that based on different statistics such as <b>term frequency</b> and <b>inverse document frequency</b>. These two models have been widely used as baseline models and adopted by industries due to their robustness and simplicity. The similarity scores between the original question and candidate questions can be calculated by different weighting scheme, and the final output is a ranked list of candidate questions.  
Here I chose to use *Indri* as the main toolkit for model implementation, and *trec_eval* as main evaluation platforms. As for parameter tuning, I used *grid_search* to find out the optimal ones.  
After getting my results, I decided to add a new advanced retrieval technique **RM3** to my two tested models. The reason is that **RM3** has been proved can enhance retrieval performance by augmenting original questions' informativeness.  
b) **Learning to rank Models (L2R)**  
This categorized retrieval models are based on machine learning appoaches that focus on learning the similar pattern between original question and it's candidate questions. The input of L2R models are a set of features of text such as term frequency, text length and similarity scores obtained from traditional models.  
Here I decided to implement 6 most commonly-used L2R models on *Ranklib* and apply default tuning method for all models. Also, I reported the average performance over five running times for each tested model.  
c) **Neural Ranking Models**  
Unlike  L2R  models  that  employ  machine  learning  techniques  over  handcrafted  features,  neuralranking models learn the features of text representation and matching patterns from the raw textautomatically by using the neural network. In my work, I chose to implement 4 short text matching neural network on *Matchzoo*. *Matchzoo* is implemented in *Keras* and employs *Tensorflow* as backend. Here, in order to avoid overfitting, epoch number is fixed to 20 during the training process and other tuning parameters are set as default. Similar to L2R models, I also reported the average performance over five running times for each tested model. 
d) **BERT**  





