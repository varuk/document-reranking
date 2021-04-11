import time
from pickle import dump,load,dumps,loads
import nltk
from nltk.corpus import stopwords
import re
import sys
import math
from krovetzstemmer import Stemmer 
stemmer = Stemmer()
swords = set(stopwords.words("english"))
Dictionary = dict()
Bicounts = dict()
Documents = dict()
Sum_of_tokens = 0
Avg_doclength = 0
N = 0
def tokenize(tokens):
    tokens = re.sub('[\'\-,\+\;<>/\!\"\(\)\{\}\?]', ' ',tokens).lower().split()
    tf= dict()
    tfbi = dict()
    doclength = 0
    prev = ""
    for token in tokens:
        if token  not in swords and not re.search('[0-9]+',token):
            token = stemmer.stem(token)
            if not prev=="":
                if token in Bicounts.keys():
                    Bicounts[token][prev] = Bicounts[token].get(prev,0) + 1
                else:
                    Bicounts[token] = {prev:1}
                if token in tfbi.keys():
                    tfbi[token][prev] = tfbi[token].get(prev,0) + 1
                else:
                    tfbi[token] = {prev:1}
            prev = token
            tf[token] = tf.get(token,0) + 1
            Dictionary[token] = Dictionary.get(token,0)+1  
            doclength = doclength + 1
    return doclength,tf, tfbi
def create_dict(file):
    global Sum_of_tokens
    global Avg_doclength
    global N
    In = open(file,'r')
    Out = open("output",'wb')
    line = In.readline()
    while line:
        doc = line.split('\t')
        text = doc[3]
        ID = doc[0]
        doclength,uni_tokens, bi_tokens = tokenize(text)
        Sum_of_tokens = Sum_of_tokens + doclength
        byteoffset = Out.tell()
        Documents[ID] = (byteoffset, doclength)
        data = dumps((uni_tokens,bi_tokens))
        Out.write(data)
        line = In.readline()
    N = len(Documents.keys())
    Avg_doclength = Sum_of_tokens/N
    In.close()
    Out.close()
def process(query):
    tokens = re.sub('[\'\-,\+\;<>/\!\"\(\)\{\}\?]', ' ',query).lower().split()
    qterms = []
    for token in tokens:
        if token not in swords and not re.search('[0-9]+',token):
            token = stemmer.stem(token)
            qterms.append(token)
    return qterms

def unilm(top100docs,query,u):
    lms = dict()
    docfile= open("output",'rb')
    Ld = dict()
    Vocab = set()
    for docid in top100docs:
        byteoffset, doclength = Documents[docid]
        Ld[docid] = doclength
        docfile.seek(byteoffset)
        tokens,_ = loads(docfile.read())
        probs = dict()
        for token,tf in tokens.items():
            Vocab.add(token)
            probs[token] = tf/doclength
        lms[docid] = probs
    docfile.close()
    Rel_model = dict()
    prob_M = 1/100
    for w in Vocab:
        prob_wq = 0
        prob_q = 1
        for docid in top100docs:
            probs = lms[docid]
            lamda = u/(u+ Ld[docid])
            prob_wq_given_m= (1-lamda)*probs.get(w,0) + (lamda)*Dictionary.get(w,0)/Sum_of_tokens
            prob_q_given_m = 1
            for token in query:
                prob_q_given_m =  prob_q_given_m*((1-lamda)*probs.get(token,0) + (lamda)*Dictionary.get(token,0)/Sum_of_tokens) 
                prob_wq_given_m = prob_wq_given_m*((1-lamda)*probs.get(token,0) + (lamda)*Dictionary.get(token,0)/Sum_of_tokens) 
            prob_wq = prob_wq + prob_wq_given_m*prob_M 
            prob_q = prob_q + prob_q_given_m*prob_M
        Rel_model[w]= prob_wq / prob_q
    scores = dict()
    for docid in top100docs: 
        V = set(lms[docid].keys())
        for w in V:
            probs = lms[docid]
            lamda = u/(u+ Ld[docid])
            prob_w_given_m = (1-lamda)*probs.get(w,0) + (lamda)*Dictionary.get(w,0)/Sum_of_tokens
            scores[docid] = scores.get(docid,0) + Rel_model[w]*math.log(prob_w_given_m/((lamda)*Dictionary.get(w,0)/Sum_of_tokens))
    answers = sorted(scores.keys(), key=lambda x: (scores[x]),reverse= True)
    return answers, scores

def bilm(top100docs,query,u,u1,u2):
    lms = dict()
    bilms = dict()
    docfile= open("output",'rb')
    Ld = dict()
    Vocab = set()
    for docid in top100docs:
        byteoffset, doclength = Documents[docid]
        Ld[docid] = doclength
        docfile.seek(byteoffset)
        unitokens, bitokens = loads(docfile.read())
        probs = dict()
        biprobs = dict()
        prev = ""
        for token,tf in unitokens.items():
            Vocab.add(token)
            probs[token] = tf/doclength
        for token in bitokens.keys():
            dict2 = bitokens[token]
            for prev, count in dict2.items():
                if token in biprobs.keys():
                    biprobs[token][prev] = count/doclength
                else:
                    biprobs[token] = {prev:count/doclength}
        lms[docid] = probs
        bilms[docid] = biprobs
    Rel_model = dict()
    prob_M = 1/100
    for w in Vocab:
        prob_wq = 0
        prob_q = 1
        for docid in top100docs:
            probs = lms[docid]
            biprobs = bilms[docid]
            lamda = u/(u+ Ld[docid])
            lamda2 = u1/(u1 + Ld[docid])
            lamda3 = u2/(u2 + Ld[docid])
            token = query[0]
            prev = token
            prob_wq_given_m= (1-lamda)*probs.get(token,0) + (lamda)*Dictionary.get(token,0)/Sum_of_tokens
            prob_q_given_m = (1-lamda)*probs.get(token,0) + (lamda)*Dictionary.get(token,0)/Sum_of_tokens
            for token in query[1:]:
                if token in biprobs.keys():
                    prod= (1-lamda)*((1-lamda2)*biprobs[token].get(prev,0)/probs.get(prev,1) + lamda2*probs[token])  + (lamda)*((1-lamda3)*Bicounts[token].get(prev,0)/Dictionary.get(prev,1) + lamda3*(Dictionary[token]/Sum_of_tokens))
                    prob_q_given_m =  prob_q_given_m*prod 
                    prob_wq_given_m = prob_wq_given_m*prod
                if token not in biprobs.keys():
                    prod = (1 -lamda)*lamda2*probs.get(token,0) + lamda*lamda3*(Dictionary.get(token,0)/Sum_of_tokens)
                    prob_q_given_m =  prob_q_given_m*prod 
                    prob_wq_given_m = prob_wq_given_m*prod
                prev = token
            prob_wq = prob_wq + prob_wq_given_m*prob_M 
            prob_q = prob_q + prob_q_given_m*prob_M
        Rel_model[w]= prob_wq / prob_q
    scores = dict()
    for docid in top100docs: 
        V = set(lms[docid].keys())
        for w in V:
            probs = lms[docid]
            lamda = u/(u+ Ld[docid])
            prob_w_given_m = (1-lamda)*probs.get(w,0) + (lamda)*Dictionary.get(w,0)/Sum_of_tokens
            scores[docid] = scores.get(docid,0) + Rel_model[w]*math.log(prob_w_given_m/((lamda)*Dictionary.get(w,0)/Sum_of_tokens))
    answers = sorted(scores.keys(), key=lambda x: (scores[x]),reverse= True)
    return answers, scores


def rerank(queryfile, top100file, model):
    outfile = open("results",'w')
    top100 = dict()
    file = open(top100file, 'r')
    line = file.readline()
    while line:
        result = line.split()
        top100[result[0]] = top100.get(result[0],[]) + [(result[2])]   
        line = file.readline()
    file.close()
    file = open(queryfile, 'r')
    line = file.readline()
    while line:
        qID,query = line.split('\t')
        query=process(query)
        relevantdocs=top100.get(qID)
        docids = []
        scores = {}
        if model=="uni":
            docids,scores = unilm(relevantdocs,query,1000)
        else:
            docids,scores = bilm(relevantdocs,query,1000,1000,1000) 
        for i in range(100):
            iter = 'Q0'
            docid = str(docids[i])
            rank = str((i +1))
            score = str(float(scores[docid]))
            outfile.write(str(int(qID))+" "+iter+" "+docid+" "+rank+" "+score+" "+ "IndriQueryLikelihood" + "\n") 
        line = file.readline()
    file.close()
    outfile.close()
if __name__=="__main__":
    collection_file = sys.argv[3]
    query_file = sys.argv[1]
    top_100_file = sys.argv[2]
    model = sys.argv[4]
    create_dict(collection_file)
    rerank(query_file, top_100_file, model)
