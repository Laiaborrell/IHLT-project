from nltk.metrics import jaccard_distance
from nltk import ngrams

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


def compare_words_ngrams(a, b, n):
	ngrams_a = ngrams(a.split(), n)
	ngrams_b = ngrams(b.split(), n)
	return jacc_sim(ngrams_a, ngrams_b)
