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
Documents = dict()
Sum_of_tokens = 0
Avg_doclength = 0
N = 0
def tokenize(tokens):
    tokens = re.sub('[\'\-,\+\;<>/\!\"\(\)\{\}\?]', ' ',tokens).lower().split()
    tf= dict()
    doclength = 0
    for token in tokens:
        if token  not in swords and not re.search('[0-9]+',token):
            token = stemmer.stem(token)
            tf[token] = tf.get(token,0) + 1
            if tf[token]==1: 
                Dictionary[token] = Dictionary.get(token,0)+1  
            doclength = doclength + 1
    return doclength,tf
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
        doclength,unique_tokens = tokenize(text)
        Sum_of_tokens = Sum_of_tokens + doclength
        byteoffset = Out.tell()
        Documents[ID] = (byteoffset, doclength)
        data = dumps(unique_tokens)
        Out.write(data)
        line = In.readline()
    N = len(Documents.keys())
    Avg_doclength = Sum_of_tokens/N
    In.close()
    Out.close()
def QEandBM25(top100docs,query,k1,b,m):
    doc_scores = {}
    term_scores = {}
    F = open("output",'rb')
    doc_lengths = {}
    doc_tfs = {}
    p = {}
    for i in range(100):
        docid = top100docs[i]
        byteoffset,doclength = Documents[docid]
        F.seek(byteoffset)
        doc_lengths[docid]  = doclength
        tfs = loads(F.read())
        doc_tfs[docid] = tfs
        terms = set(tfs.keys())
        for term in terms:
            p[term] = p.get(term,0)+1
    All_terms = p.keys()
    for term in All_terms:
        df = Dictionary[term]
        ui = (df+.5)/N
        pi = (p[term]+.5)/101
        wt = math.log(pi/(1-pi)) - math.log(ui/(1-ui))
        term_scores[term] = wt*p[term]
    top_terms = sorted(term_scores.keys(), key=lambda x: (term_scores[x]),reverse= True)
    queryterms = set(query.keys())
    i = 0
    for term in top_terms:
        if i==m:
            break
        if term not in queryterms:
            query[term] = 1
            i = i +1
    for q,qfi in query.items():
        df = Dictionary[q]
        ui = df/N
        pi = (p.get(q,0) + .5)/101
        w = math.log(pi/(1-pi)) - math.log(ui/(1-ui))
        for docid in top100docs:
            dli = doc_lengths[docid]
            tf = doc_tfs[docid].get(q,0)
            doc_scores[docid] = doc_scores.get(docid,0) + ((qfi*tf*(1+k1))/(k1*(1-b + b*dli/Avg_doclength) + tf))*w
    reranked_documents = sorted(doc_scores.keys(), key=lambda x: (doc_scores[x]),reverse= True)
    return reranked_documents,doc_scores

def process(query):
    tokens = re.sub('[\'\-,\+\;<>/\!\"\(\)\{\}\?]', ' ',query).lower().split()
    qf = dict()
    for token in tokens:
        if token not in swords and not re.search('[0-9]+',token):
            token = stemmer.stem(token)
            qf[token] = qf.get(token, 0) +1
    return qf

def rerank(queryfile, top100file, m):
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
        #print(query)
        query=process(query)
        relevantdocs=top100.get(qID)
        docids,scores = QEandBM25(relevantdocs,query, .9,.4,m)
        for i in range(100):
            iter = 'Q0'
            docid = str(docids[i])
            rank = str(i)
            score = str(float(scores[docid]))
            outfile.write(str(int(qID))+" "+iter+" "+docid+" "+rank+" "+score+" "+ "IndriQueryLikelihood" + "\n") 
        line = file.readline()
    file.close()
    outfile.close()
if __name__=="__main__":
    collection_file = sys.argv[3]
    query_file = sys.argv[1]
    top_100_file = sys.argv[2]
    expansion_limit = int(sys.argv[4])
    create_dict(collection_file)
    rerank(query_file, top_100_file, expansion_limit)
