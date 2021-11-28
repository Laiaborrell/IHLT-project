import nltk
from nltk.metrics import jaccard_distance
from nltk import ngrams
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet_ic')
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.stem import WordNetLemmatizer
import numpy as np
import spacy
nltk.download('stopwords')


#STRING BASED MEASURES
def lc_substring(a, b):
    m = len(a)
    n = len(b)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(a[i-c+1:i+1])
                elif c == longest:
                    lcs_set.add(a[i-c+1:i+1])

    return len(lcs_set)


def lc_subsequence(S1, S2):
	m = len(S1)
	n = len(S2)

	L = [[0 for x in range(n+1)] for x in range(m+1)]

    # Building the mtrix in bottom-up way
	for i in range(m+1):
		for j in range(n+1):
			if i == 0 or j == 0:
				L[i][j] = 0
			elif S1[i-1] == S2[j-1]:
				L[i][j] = L[i-1][j-1] + 1
			else:
				L[i][j] = max(L[i-1][j], L[i][j-1])
	index = L[m][n]
	lcs = [""] * (index+1)
	lcs[index] = ""
	i = m
	j = n
	while i > 0 and j > 0:
		if S1[i-1] == S2[j-1]:
			lcs[index-1] = S1[i-1]
			i -= 1
			j -= 1
			index -= 1
		elif L[i-1][j] > L[i][j-1]:
			i -= 1
		else:
			j -= 1
			
	return len(lcs)


def jacc_sim(list1, list2):
	return 1-jaccard_distance(set(list1), set(list2)) # 1-distance


def get_character_ngrams(sentence, n):
	return [sentence[i:i+n] for i in range(len(sentence)-n+1)]


def compare_character_ngrams(a, b, n):
	ngrams_a = get_character_ngrams(a, n)
	ngrams_b = get_character_ngrams(b, n)
	return jacc_sim(ngrams_a, ngrams_b)


#aquest de compare words potser hauria d'anar a semantic similarity measures?
def compare_words_ngrams(a, b, n):
	ngrams_a = ngrams(a, n)
	ngrams_b = ngrams(b, n)

	ngrams_final_a = []
	for i in ngrams_a:
		ngrams_final_a.append(i)
	ngrams_final_b = []
	for i in ngrams_b:
		ngrams_final_b.append(i)

	if ngrams_final_a == [] or ngrams_final_b == []:
		return 1
	
	return jacc_sim(ngrams_final_a, ngrams_final_b)

def compare_postag_ngrams(a, b, n):
	postags_ngrams_a = []
	postags_ngrams_b = []
	a_pos = nltk.pos_tag(a)
	b_pos = nltk.pos_tag(b)
	for pair in a_pos:
		postags_ngrams_a.append(pair[1])
	for pair in b_pos:
		postags_ngrams_b.append(pair[1])

	return compare_words_ngrams(postags_ngrams_a, postags_ngrams_b, n)


#SEMANTIC SIMILARITY MEASURES
def lemmatize_list(wordsList):
	wnl = nltk.stem.WordNetLemmatizer()
	pairs = nltk.pos_tag(wordsList)  # pairs of word and its pos
	lemmatizedList = []
	for pair in pairs:
		if pair[1][0] in {'N', 'V'}:  # si es nom o verb
			lemmatizedList.append(wnl.lemmatize(pair[0].lower(), pos=pair[1][0].lower()))
		else:
			lemmatizedList.append(pair[0])
	return lemmatizedList

def lemmas_similarity(a,b):
	a_lem, b_lem = lemmatize_list(a), lemmatize_list(b)
	return jacc_sim(a_lem, b_lem)

def lemmatize_word(word, pos):
	wnl = WordNetLemmatizer()
	if pos in {'n', 'v'}:
		return wnl.lemmatize(word, pos=pos)
	return word  # if no lemma, return original word

def get_synsets(words_pos_pairs):
	synsets = []
	categories = []
	for pair in words_pos_pairs:
		word,pos = pair
		pos = pos[0].lower()
		word_lemma = lemmatize_word(word,pos)
		if pos in ['n','v']:
			try:
				synset = wn.synset(f"{word_lemma}.{pos}.01")
			except:
				synset = None
		else:
			synset = None
		synsets.append(synset)
		categories.append(pos)
	return synsets,categories

brown_ic = wordnet_ic.ic('ic-brown.dat')

def path_similarity(a,b):
	pairs_a = nltk.pos_tag(a)
	pairs_b = nltk.pos_tag(b)
	syns_a, categories_a = get_synsets(pairs_a)
	syns_b, categories_b = get_synsets(pairs_b)
	similarity=[]
	for s1,c1 in zip(syns_a,categories_a):
		for s2,c2 in zip(syns_b,categories_b):
			if c1==c2 and s1 and s2:
				similarity.append(s1.path_similarity(s2, brown_ic))
	if len(similarity)!=0:
		max_sim = np.max(similarity)
	else:
		max_sim = 0
	return max_sim #retornem la similitud mes gran entre dos synsets de les frases


def retokenize_and_stack(x):
	with x.retokenize() as retokenizer:
		tokens = [token for token in x]
		for ent in x.ents:
			retokenizer.merge(x[ent.start:ent.end],
							  attrs={"LEMMA": " ".join([tokens[i].text for i in range(ent.start, ent.end)])})
	texts = []
	for token in x:
		texts.append(token.text)
	return texts

def words_NE_similarity(nlp, sent_a,sent_b):
	x_a, x_b = (nlp(sent_a), nlp(sent_b))
	list_a, list_b = (retokenize_and_stack(x_a), retokenize_and_stack(x_b))
	return jacc_sim(list_a,list_b)

def WSD(a,b): #word sense desambiguation
	postags = {"V": 'v', "N": 'n', "J": 'a', "R": 'r'}
	pairs_a,pairs_b = (nltk.pos_tag(a),nltk.pos_tag(b))
	synsetsList_a,synsetsList_b = ([],[])
	for pair_a in pairs_a:
		if pair_a[1][0] in postags.keys():  # pair[1] es el pos
			synsetsList_a.append(nltk.wsd.lesk(a, pair_a[0], postags[pair_a[1][0]]))
		else:
			continue
	for pair_b in pairs_b:
		if pair_b[1][0] in postags.keys():  # pair[1] es el pos
			synsetsList_b.append(nltk.wsd.lesk(a, pair_b[0], postags[pair_b[1][0]]))
		else:
			continue
	clean_synsetsList_a = []
	clean_synsetsList_b = []
	for s in synsetsList_a:
		if str(s) != 'None':
			clean_synsetsList_a.append(s)
	for s in synsetsList_b:
		if str(s) != 'None':
			clean_synsetsList_b.append(s)
	return jacc_sim(set(clean_synsetsList_a), set(clean_synsetsList_b))

def stopWordsFilter(sw, words):
	toRemove = ['.', ',', ';', ':', '\'', '"', '$', '#', '@', '!', '?', '/', '*', '&', '^', '-', '+','."'] #punctuation marks
	aw = ['thou', 'thee', 'thy', 'er'] #archaic words to remove
	remove_this = []

	for i, word in enumerate(words):
		wordl = word.lower()
		if wordl in toRemove or wordl in aw or wordl in sw:
			remove_this.append(i)
	
	for idx, i in enumerate(remove_this):
		words.pop(i - idx)
	
	return words

#GET METRICS
def get_metrics(dt,lexical=False):
	r, _ = dt.shape
	metrics = {}
	a_words, b_words, js_w = [], [], []
	a_words_wo_stop, b_words_wo_stop, js_w_wo_stop = [], [], []
	sw = set(nltk.corpus.stopwords.words('english'))
	#nlp = spacy.load("en_core_web_sm")

	for i in range(r): # iteration over the rows
		sent_a,sent_b = (dt[0][i],dt[1][i])
		words_a, words_b = (nltk.word_tokenize(sent_a) , nltk.word_tokenize(sent_b))
		a_words.append(words_a)
		b_words.append(words_b)
		js_w.append(jacc_sim(words_a, words_b))
		words_wo_stop_a, words_wo_stop_b = (stopWordsFilter(sw, words_a), stopWordsFilter(sw, words_b))
		a_words_wo_stop.append(words_wo_stop_a)
		b_words_wo_stop.append(words_wo_stop_b)
		js_w_wo_stop.append(jacc_sim(words_wo_stop_a, words_wo_stop_b))

	dt['words_a'] = a_words
	dt['words_b'] = b_words
	metrics['words_js'] = js_w
	dt['words_a_wo_stop'] = a_words_wo_stop
	dt['words_b_wo_stop'] = b_words_wo_stop
	metrics['words_wo_stop_js'] = js_w_wo_stop

	# Initializing metric lists
	c_ngrams_n = 8
	w_ngrams_n = 8
	postag_n = 8
    

	for i in range(1,c_ngrams_n):
		c_metric_name = 'c_ngrams_'+str(i)
		metrics[c_metric_name] = []

	for i in range(1,w_ngrams_n):
		w_metric_name = 'w_ngrams_'+str(i)
		metrics[w_metric_name] = []
		w_metric_name = 'w_ngrams_wo_stop'+str(i)
		metrics[w_metric_name] = []
	if lexical==False:
		for i in range(1,postag_n):
			postag_metric_name = 'postag_ngrams_'+str(i)
			metrics[postag_metric_name] = []
			postag_metric_name = 'postag_ngrams_wo_stop'+str(i)
			metrics[postag_metric_name] = []

	metrics['lc_substring']	= []
	metrics['lc_subsequence'] = []
	metrics['path_s'] = []
	metrics['path_s_wo_stop'] = []
	metrics['lemm_jc_s'] = []
	metrics['lemm_jc_s_wo_stop'] = []
	#metrics['wordsNE_jc_s'] = []
	metrics['WSD_jc_s'] = []
	metrics['WSD_jc_s_wo_stop'] = []


	for i in range(r): # Metrics loop
		metrics['lc_substring'].append(lc_substring(dt[0][i], dt[1][i]))
		metrics['lc_subsequence'].append(lc_subsequence(dt[0][i], dt[1][i]))
		metrics['path_s'].append(path_similarity(dt['words_a'][i], dt['words_b'][i]))
		metrics['path_s_wo_stop'].append(path_similarity(dt['words_a_wo_stop'][i], dt['words_b_wo_stop'][i]))
		metrics['lemm_jc_s'].append(lemmas_similarity(dt['words_a'][i], dt['words_b'][i]))
		metrics['lemm_jc_s_wo_stop'].append(lemmas_similarity(dt['words_a_wo_stop'][i], dt['words_b_wo_stop'][i]))
		#metrics['wordsNE_jc_s'].append(words_NE_similarity(nlp, dt[0][i], dt[1][i]))
		metrics['WSD_jc_s'].append(WSD(dt['words_a'][i], dt['words_b'][i]))
		metrics['WSD_jc_s_wo_stop'].append(WSD(dt['words_a_wo_stop'][i], dt['words_b_wo_stop'][i]))

		for k in range(1,c_ngrams_n):
			c_metric_name = 'c_ngrams_'+str(k)
			metrics[c_metric_name].append(compare_character_ngrams(dt[0][i], dt[1][i], k))
		for k in range(1, w_ngrams_n):
			w_metric_name = 'w_ngrams_wo_stop'+str(k)
			metrics[w_metric_name].append(compare_words_ngrams(dt['words_a_wo_stop'][i], dt['words_b_wo_stop'][i], k))
			w_metric_name = 'w_ngrams_'+str(k)
			metrics[w_metric_name].append(compare_words_ngrams(dt['words_a'][i], dt['words_b'][i], k))
		if lexical==False:
			for k in range(1,postag_n):
				postag_metric_name = 'postag_ngrams_wo_stop'+str(k)
				metrics[postag_metric_name].append(compare_postag_ngrams(dt['words_a_wo_stop'][i], dt['words_b_wo_stop'][i], k))
				postag_metric_name = 'postag_ngrams_'+str(k)
				metrics[postag_metric_name].append(compare_postag_ngrams(dt['words_a'][i], dt['words_b'][i], k))
	return metrics

def get_syntactic_metrics(dt,syntactic=False):
	r, _ = dt.shape
	metrics = {}
	a_words, b_words = [], []
	a_words_wo_stop, b_words_wo_stop= [], []
	sw = set(nltk.corpus.stopwords.words('english'))

	for i in range(r): # iteration over the rows
		sent_a,sent_b = (dt[0][i],dt[1][i])
		words_a, words_b = (nltk.word_tokenize(sent_a) , nltk.word_tokenize(sent_b))
		a_words.append(words_a)
		b_words.append(words_b)
		words_wo_stop_a, words_wo_stop_b = (stopWordsFilter(sw, words_a), stopWordsFilter(sw, words_b))
		a_words_wo_stop.append(words_wo_stop_a)
		b_words_wo_stop.append(words_wo_stop_b)

	dt['words_a'] = a_words
	dt['words_b'] = b_words
	dt['words_a_wo_stop'] = a_words_wo_stop
	dt['words_b_wo_stop'] = b_words_wo_stop

	# Initializing metric lists
	postag_n = 8

	for i in range(1, postag_n):
		postag_metric_name = 'postag_ngrams_'+str(i)
		metrics[postag_metric_name] = []
		postag_metric_name = 'postag_ngrams_wo_stop'+str(i)
		metrics[postag_metric_name] = []
        
	for i in range(r): # Metrics loop
		for k in range(1,postag_n):
 			postag_metric_name = 'postag_ngrams_wo_stop'+str(k)
 			metrics[postag_metric_name].append(compare_postag_ngrams(dt['words_a_wo_stop'][i], dt['words_b_wo_stop'][i], k))
 			postag_metric_name = 'postag_ngrams_'+str(k)
 			metrics[postag_metric_name].append(compare_postag_ngrams(dt['words_a'][i], dt['words_b'][i], k))
	return metrics