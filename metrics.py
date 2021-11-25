import nltk
from nltk.metrics import jaccard_distance
from nltk import ngrams
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import treetaggerwrapper as ttpw

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
	ngrams_a = ngrams(a.split(), n)
	ngrams_b = ngrams(b.split(), n)

	ngrams_final_a = []
	for i in ngrams_a:
		ngrams_final_a.append(i)
	ngrams_final_b = []
	for i in ngrams_b:
		ngrams_final_b.append(i)

	if ngrams_final_a == [] or ngrams_final_b == []:
		return 1
	
	return jacc_sim(ngrams_final_a, ngrams_final_b)


#SEMANTIC SIMILARITY MEASURES
def get_synsets(words_pos_pairs):
	PoS_to_WN = {
		"NN": "n",
		"VB": "v",
		"DT": None,
		"PR": None,
		"CC": None
	}
	synsets = []
	for pair in words_pos_pairs:
		word,pos = pair
		if PoS_to_WN[pos]!=None:
			synset = wn.synset(f"{word}.{PoS_to_WN[pos]}.01")
			if synset != None: #si té synset
				synsets.append(synset)
	return synsets

def resnik_similarity(a,b):
	brown_ic = wordnet_ic.ic('ic-brown.dat')
	pairs_a = nltk.pos_tag(a)
	pairs_b = nltk.pos_tag(b)
	syns_a = get_synsets(pairs_a)
	syns_b = get_synsets(pairs_b)

	similarity=[]
	for s1 in syns_a:
		for s2 in syns_b:
			similarity.append(s1.res_similarity(s2, brown_ic))
	max_sim = np.max(similarity)
	return max_sim #retornem la similitud mes gran entre dos synsets de les frases


def lemmatize(tagger, text):
    tags = tagger.tag_text(text)
    lemmas = [t.split('\t')[-1] for t in tags]
    return lemmas

def get_metrics(dt):
	r, _ = dt.shape
	metrics = {}

	#tagger = ttpw.TreeTagger(TAGLANG='en')
	a_lems, b_lems, js_l = [], [], []
	a_words, b_words, js_w = [], [], []
	
	for i in range(r): # iteration over the rows
		words_a, words_b = (nltk.word_tokenize(dt[0][i]) , nltk.word_tokenize(dt[1][i]))
		#lemmas_a, lemmas_b = (lemmatize(tagger, dt[0][i]) , lemmatize(tagger, dt[1][i]))
		#a_lems.append(lemmas_a)
		#b_lems.append(lemmas_b)
		a_words.append(words_a)
		b_words.append(words_b)
		#js_l = jacc_sim(lemmas_a, lemmas_b)
		js_w.append(jacc_sim(words_a, words_b))

	dt['words_a'] = a_words
	dt['words_b'] = b_words
	#dt['lemmas_a'] = a_lems
	#dt['lemmas_b'] = b_lems
	#metrics['lemmas_js'] = js_l
	metrics['words_js'] = js_w

	# Initializing metric lists
	c_ngrams_n = 4
	w_ngrams_n = 4

	for i in range(1,c_ngrams_n):
		c_metric_name = 'c_ngrams_'+str(i)
		metrics[c_metric_name] = []
	for i in range(1,w_ngrams_n):
		w_metric_name = 'w_ngrams_'+str(i)
		metrics[w_metric_name] = []
	metrics['lc_substring']	= []
	metrics['lc_subsequence'] = []
	metrics['resnik_s'] = []

	for i in range(r): # Metrics loop
		metrics['lc_substring'].append(lc_substring(dt[0][i], dt[1][i]))
		metrics['lc_subsequence'].append(lc_subsequence(dt[0][i], dt[1][i]))
		for k in range(1,c_ngrams_n):
			c_metric_name = 'c_ngrams_'+str(k)
			metrics[c_metric_name].append(compare_character_ngrams(dt[0][i], dt[1][i], k))
		for k in range(1, w_ngrams_n):
			w_metric_name = 'w_ngrams_'+str(k)
			metrics[w_metric_name].append(compare_words_ngrams(dt[0][i], dt[1][i], k))
		metrics['resnik_s'].append(resnik_similarity(dt['words_a'][i], dt['words_b'][i]))


	return metrics