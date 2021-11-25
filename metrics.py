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
	return jacc_sim(ngrams_a, ngrams_b)

#SEMANTIC SIMILARITY MEASURES

PoS_to_WN = {
    "NN": "n",
    "VB": "v",
    "DT": None,
    "PR": None,
    "CC": None
}

def get_synsets(words_pos_pairs):
	synsets = []
	categories = []
	for pair in words_pos_pairs:
		word,pos = pair
		if PoS_to_WN[pos]!=None:
			synset = wn.synset(f"{word}.{PoS_to_WN[pos]}.01")
			if synset != None: #si té synset
				synsets.append(synset)
				categories.append(PoS_to_WN[pos])
	return synsets,categories

def resnik_similarity(a,b):
	#hem de treure allò dels synsets, però aquest cop no tenim els postaggers, se viene
	pairs_a = nltk.pos_tag(a)
	pairs_b = nltk.pos_tag(b)
	syns_a = get_synsets(pairs_a)
	syns_b = get_synsets(pairs_b)

	#sets de synsets pero millor agafar la similitud mes gran entre dos sinsets de les frases

	brown_ic = wordnet_ic.ic('ic-brown.dat') #no cal que ho generi cada cop, treureho de aqui
	try:
		sim = syn1.res_similarity(syn2, brown_ic)
	except:
		sim = 0
	#print("Lin Similarity between {} and {} = {} \n".format(word1, word2, sim))
	return sim

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
		js_w = jacc_sim(words_a, words_b)

	dt['words_a'] = a_words
	dt['words_b'] = b_words
	#dt['lemmas_a'] = a_lems
	#dt['lemmas_b'] = b_lems
	metrics['lemmas_js'] = js_l
	metrics['words_js'] = js_w

	# Metrics loop
	for i in range(r):
		metrics['lc_substring'] = lc_substring(dt[0][i], dt[1][i])
		metrics['lc_subsequence'] = lc_subsequence(dt[0][i], dt[1][i])
		for j in range(1,4):
			w_metric_name = 'w_ngrams_'+str(i)
			c_metric_name = 'c_ngrams_'+str(i)
			metrics[w_metric_name] = compare_words_ngrams(dt[0][i], dt[1][i], j)
			metrics[c_metric_name] = compare_character_ngrams(dt[0][i], dt[1][i], j)

	return metrics