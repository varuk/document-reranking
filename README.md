# document-reranking
This code was written in python=3.7.6 and is comptabile with python < 3.9.0 but only python 3.x. In addition to the standard library packages of python, this project uses other packages mentioned in requirements.txt.
To install, use 'make' command on terminal. 
The re-ranked documents file which is created when the documents are called, is named as "output."
The task is, given 100 documents for each query, implement probabilistic retrieval query expansion + bm25 as well as relevance model based language modelling to re rank documents. 
prob_rerank.py contains the query expansion model. lm_rerank.py contains language models. 
