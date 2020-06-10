import string
from os import walk
import json
import numpy as np
from math import inf
from math import sqrt,ceil
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import tkinter as tk
import sys, os
import string
import collections
from nltk.stem import WordNetLemmatizer
from collections import Counter 
from numpy import ndarray
global dirname
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from spellchecker import SpellChecker
import SemanticSimilarity as ss
dirname="data"
r_flag=0

def extract_citation_sentence():
    try:
        citation_sentence = []
        numeric_flag = 0
        #load files in chosen directory
        files=[]
        
        if os.path.isdir(dirname) == False:
            error_msg = "The directory for indexing does not exist, please enter different path in the code line 22"
            print(error_msg)
            l2.configure(text=error_msg)
            
        
        for dirpath,dirnames,filenames in walk(dirname):  
            for j in filenames:
                files.append(j)

        for file in files:

            f = open(dirname+'/'+file,'r')

            x = f.read()
            for sentence in x.split("."):
                for i in range(len(sentence)):
                    if sentence[i] == '[':
                        for j in range(1,5):
                            if i + j >= len(sentence):
                                break
                            if sentence[i+j].isnumeric():
                                numeric_flag = 1
                            elif sentence[i+j] == ']' and numeric_flag == 1:
                                numeric_flag = 0
                                citation_sentence.append(sentence[:i]+sentence[i+j+1:])
                                break
                            else:
                                break
                        if(numeric_flag == 0):
                            break
            f.close()

        return citation_sentence
    
    except Exception as e:
        print("in extract_citation_sentence")
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def read_cache():
    try:
        #loading saved compressed dictionary of numpy arrays
        loaded = np.load(dirname+'.npz',allow_pickle=True)
        return 1,loaded
    except Exception as e:
        print("in read cache")
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return 0,None
def write_cache(to_save):
    try:
        #saving compressed dictionary of numpy arrays 
        np.savez_compressed(dirname,terms=np.array(to_save['terms']),weight=to_save['weight'],tf=to_save['tf'],idf=to_save['idf'],df=to_save['df'],doc_norm=to_save['doc_norm'],doc_length=to_save['doc_length'],citation_sentences=to_save['citation_sentences'])
    except Exception as e:
        print("in write cache")
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def simplify(sentence):
    #remove irrelevent symbols from sentence
    exclude = set(string.punctuation)
    sentence = ''.join(ch.lower() for ch in sentence if ch not in exclude and not ch.isnumeric())
    return sentence

def cosine_similarity(x,y):
    #sim (x,y) = x.y / |x|*|y|
    x=np.array(x)
    y=np.array(y)
    x_norm=np.linalg.norm(x)
    y_norm=np.linalg.norm(y)
    result=(x*y)/(x_norm*y_norm)
    return np.sum(result)

def preprocessing(word):
    try:
        lemma = WordNetLemmatizer()
        return lemma.lemmatize((lemma.lemmatize(word.lower(), pos = 'n')), pos = 'v')
    except Exception as e:
        print("in preprocessing")
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return None

def reindex():
    try:
        
        if os.path.isdir(dirname) == False:
            error_msg = "The directory for indexing does not exist, please enter different path in the code line 22"
            print(error_msg)
            l2.configure(text=error_msg)
            return
            
        os.remove(dirname+'.npz')
        indexing()
    except Exception as e:
        print("in preprocessing")
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        indexing()

def indexing():
    global dirname
    global r_flag
    try:
        print("Processing...")
        doc_number =-1
        N=0
        to_save={}
        
        #load saved dictionary
        read_flag, to_save = read_cache()
        
        if(read_flag==1):
            return to_save, len(to_save['doc_norm'])
        
        citation_sentences = extract_citation_sentence()
        N = len(citation_sentences)
        simplified_citation_sentences = []
        
        #save every term from every document/ citation_sentence
        terms2d=[[] for i in range(N)]
        for sentences in citation_sentences:           
            doc_number=doc_number+1
            sentence=simplify(sentences)
            simplified_citation_sentences.append(sentence)
            for word in sentence.split():
                #preprocess the word
                new_word=preprocessing(word)
                terms2d[doc_number].append(new_word)
        
        #pad the terms array to perfect square size to vectorize the operations
        pad = len(max(terms2d, key=len))
        terms=np.array([i + ["-1marker"]*(pad-len(i)) for i in terms2d])
        terms=np.reshape(terms,-1)
        terms=terms[terms != "-1marker"]
        terms=np.array(list(set(terms)))
        
        #initialize term frequency for every document and term
        tf=[[0 for i in range(len(terms))] for i in range(N)]
        for doc_no in range(len(terms2d)):
            for word in terms2d[doc_no]:
                term_index=list(terms).index(word)
                tf[doc_no][term_index]=tf[doc_no][term_index]+1
                
        tf=np.array(tf)
        #calculate length of document
        doc_length=np.sum(tf,axis=1)
        #calculate the length of documents term t appears in
        df=np.sum(tf.astype(bool),axis=0)
        #calculate idf
        idf=np.log10(N/df)
        #calculate weight by idf* normalized tf
        weight=(tf/np.reshape(doc_length,(len(doc_length),1)))*idf
        #Transpose of weight
        weight=np.reshape(weight,(N,-1))
        #calculate doc_norm
        doc_norm=np.linalg.norm(weight,axis=1)
        
        #save the dictionary of numpy arrays
        to_save={}
        if(read_flag==0):
            to_save['terms']=terms
            to_save['tf']=tf
            to_save['df']=df
            to_save['idf']=idf
            to_save['doc_length']=doc_length
            to_save['doc_norm']=doc_norm
            to_save['weight']=weight
            to_save['citation_sentences']=simplified_citation_sentences
            write_cache(to_save)
        return to_save,len(doc_norm)
    
    except Exception as e:
        print("in indexing")
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def process_query(query):
    try:
        global labelFrame
        #index the files in chosen directory
        saved,N=indexing()
        #load the saved numpy arrays
        terms=saved['terms']
        idf=saved['idf']
        weight=saved['weight']
        tf=saved['tf']
        df=saved['df']
        doc_length=saved['doc_length']
        citation_sentences=saved['citation_sentences']
        
        #return if no query is entered
        if(len(query)==0):
            return
        #create vector of term frequency of  query words
        tfq=np.zeros(len(terms))
        weight_q=np.zeros(len(terms))
        query_words=[]
        query_simplified=simplify(query)
        for words in query_simplified.split():
            new_word=preprocessing(words)
            query_words.append(new_word)
            if new_word in terms:
                index=list(terms).index(new_word)
                tfq[index]=tfq[index]+1
                
        #calculate weight of query
        weight_q = (tfq / np.sum(tfq) ) * idf
        weight_q = np.reshape(weight_q,(1,-1))
        scores = np.zeros(N)
        t_count = 0
        
        for i in range(N):
            temp = np.reshape( weight[i], (1, -1))
            #call similarity function to get score
            if cosine_similarity(weight_q, temp) > 0:
                scores[i] = cosine_similarity(weight_q, temp)
        
        score_dict = {}
        for i in range(N):
            if scores[i] > 0:
                score_dict[i] = scores[i]
        
        sorted_citation_sentences = []
        sorted_citation_sentences_weight = []
        sorted_citation_sentences_index = []
        sorted_citation_sentences_score = []
        
        #sort the scores and get the corresponding documents
        for j in sorted(score_dict, key = score_dict.get, reverse=True):
            sorted_citation_sentences.append(citation_sentences[j])
            sorted_citation_sentences_weight.append(weight[j])
            sorted_citation_sentences_index.append(j)
            sorted_citation_sentences_score.append(score_dict[j])
        
        return sorted_citation_sentences, sorted_citation_sentences_weight, sorted_citation_sentences_index, sorted_citation_sentences_score, terms
        
    except Exception as e:
        print("in process query")
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def optimal_clusters(w):
    sil = []
    kmax = len(w)-1

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax+1):
        kmeans = KMeans(n_clusters = k).fit(w)
        labels = kmeans.labels_
        sil.append(silhouette_score(w, labels, metric = 'cosine'))
        
    return sil.index(max(sil))+2

def clustering():
    try:
        query = str(e1.get())
        if(len(query) < 2 or query == None):
            return
        cs, w, i ,s, t = process_query(query)
        
        if(len(i)<=1): 
            error_msg = "No related sentence is found, please enter another query term/s, if problem stays, please re-index"
            print(error_msg)
            l2.configure(text=error_msg)
            raise "none"
            
        num_clusters = optimal_clusters(w)
        km = KMeans(n_clusters=num_clusters)
        km.fit(w)
        clusters = km.labels_.tolist()

        cluster_keywords = []

        sum_ = [[0 for i in range(len(t))] for x in range(num_clusters)]
        for x in clusters:
            w_ = list(l for l in range(len(clusters)) if clusters[l] == x )
            for y in w_:
                sum_[x] += w[y] 
            cluster_keywords.append(t[list(sum_[x]).index(max(sum_[x]))])
            
        return clusters, cluster_keywords, cs ,w, i, s, t
        
    except Exception as e:
        print("in clustering")
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def merge_cluster(clusters, cluster_keywords ):
    try:
        for x in range(len(clusters)):
            for y in range(len(clusters)):
                if x == y or clusters[x] == clusters[y]:
                    continue
                sim = ss.semanticSimilarity(cluster_keywords[x],cluster_keywords[y])
                if(sim > 0.25):
                    clusters[y] = clusters[x]
                    cluster_keywords[y] = cluster_keywords[x]
        return clusters, cluster_keywords
    except Exception as e:
        print("in merge cluster")
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def cluster_minimized( clusters):
    try:
        cluster_sorted = sorted(clusters)
        cluster_new    = [0 for x in range(len(cluster_sorted))]
        cluster_replaced    = [0 for x in range(len(cluster_sorted))]

        for x in range(1,len(cluster_new)):
            if cluster_sorted[x] > cluster_sorted[x-1]:
                cluster_new[x] = cluster_new[x-1]+1
            else:
                cluster_new[x] = cluster_new[x-1]

        for x in range(len(clusters)):
            cluster_replaced[x] = cluster_new[cluster_sorted.index( clusters[x] ) ]
        clusters=cluster_replaced

        return clusters
    except Exception as e:
        print("in cluster minimized")
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def generate_summary(event=""):
    try:
        clusters, cluster_keywords, cs ,w, i, s, t = clustering()

        clusters, cluster_keywords = merge_cluster( clusters, cluster_keywords)
        clusters = cluster_minimized( clusters )

        merged_vectors = [[] for x in range(len(set(clusters)))]
        merged_vectors_index = [[] for x in range(len(set(clusters)))]

        for x in range(len(clusters)):
            merged_vectors[clusters[x]].append(w[x])
            merged_vectors_index[clusters[x]].append(x)

        summary = []
        
        for x in range(len(merged_vectors)):
            sum_ = 0
            score_dict = {}
            for y in range(len(merged_vectors[x])):
                score_dict[merged_vectors_index[x][y]] =  np.sum(merged_vectors[x][y])

            for j in sorted(score_dict, key=score_dict.get, reverse=True)[:3]:
                summary.append(cs[j].replace('\n',' '))

        spell = SpellChecker()
        new_summary=[]

        summary = '. '.join(summary)
        for words in summary.split():
            new_summary.append(spell.correction(words))

        print(" ".join(new_summary))
        l2.configure(text=new_summary)

    except Exception as e:
        error_msg = "No related sentence is found, please enter another query term/s, if problem stays, please re-index"
        print(error_msg)
        l2.configure(text=error_msg)
        print("in generate summary")
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


try:
    master = Tk()
    master.title("CitationAS Demo")
    master.minsize(500, 400)

    labelFrame=ttk.LabelFrame(master,text="")
    labelFrame.pack(fill=X)

    b0 = Button(labelFrame, text = "Re-Index",command=reindex, bg ="black",fg="light blue",height=1)
    b0.pack(side=LEFT)

    l = Label(labelFrame,text="Summary Generator", bg = "green" , fg="white",height=1)
    l.pack(fill=BOTH, expand=True)

    labelFrame2=ttk.LabelFrame(master,text="")
    labelFrame2.pack(fill=X)

    master.bind('<Return>', None)

    b1 = Button(labelFrame2, text = "Run Query",command=generate_summary, bg ="black",fg="light blue")
    b1.grid(row=4,column=5)

    master.bind('<Return>', generate_summary)

    Label(labelFrame2,text="Input").grid(row = 4,column = 0, padx=10,pady=20, sticky="w")

    e1 = Entry(labelFrame2)
    e1.grid(row = 4,column = 3, padx=10,pady=20 ,sticky="nsew")
    e1.focus()

    labelFrame2.grid_rowconfigure(4, weight=10)
    labelFrame2.grid_columnconfigure(3, weight=10)

    size_x = 500

    labelFrame3=ttk.LabelFrame(master,text="")
    labelFrame3.pack(fill=X)

    Label(labelFrame3,text="Output").grid(row = 4,column = 0, padx=5,pady=20, sticky="w")

    l2 = Label(labelFrame3,text="",wraplength=size_x)
    l2.grid(row = 4,column = 3, padx=10,pady=20 ,sticky="w")


    labelFrame3.grid_rowconfigure(4, weight=1)
    labelFrame3.grid_columnconfigure(3, weight=1)

    mainloop()

except Exception as e:
    print("in Main")
    print(e)
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)