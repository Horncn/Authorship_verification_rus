from sklearn.feature_extraction.text import TfidfVectorizer
import json
from scipy import spatial
from collections import Counter, OrderedDict


class Doc2Data:
    # class to process one text
    # get path to text, syntagrus model, list of stop_words, and path to save json file
    # returns nothing
    # parse text and save everything in json
    def __init__(self, filepath, model, stop_words, folder):
        self.folder = folder
        if not isinstance(filepath, str):
            raise FileNotFoundError
        try:
            self.text = open(filepath, 'r', encoding='utf8').read()
        except FileNotFoundError:
            raise FileNotFoundError
        self.text = self.text.replace('\n', '').lower()
        self.filename = filepath[filepath.find('/') + 1: filepath.find('.')]  # extract file id from filename
        self.model = model  # syntagrus
        self.stop_words = stop_words
        # words_parts and deprel in dict formats to save order of values
        self.word_parts_dist = {'ADJ': 0, 'ADV': 0, 'INTJ': 0, 'NOUN': 0, 'PROPN': 0,
                                'VERB': 0, 'ADP': 0, 'AUX': 0, 'CCONJ': 0,
                                'DET': 0, 'NUM': 0, 'PART': 0, 'PRON': 0,
                                'SCONJ': 0, 'PUNCT': 0, 'SYM': 0, '<root>': 0, 'X': 0}

        self.deprel_dist = {'nsubj': 0, 'obj': 0, 'iobj': 0, 'csubj': 0,
                            'ccomp': 0, 'xcomp': 0, 'obl': 0, 'advcl': 0, 'advmod': 0,
                            'nmod': 0, 'appos': 0, 'nummod': 0, 'acl': 0,
                            'amod': 0, 'det': 0, 'case': 0, 'conj': 0, 'cc': 0,
                            'fixed': 0, 'flat': 0, 'compound': 0, 'aux': 0, 'cop': 0, 'mark': 0,
                            'punct': 0, 'root': 0, 'UNK': 0}
        self.punctuation_dist = {':': 0, ';': 0, '.': 0, '?': 0, '!': 0, '"': 0}
        self.cases_dist = {'Nom': 0, 'Gen': 0, 'Dat': 0, 'Acc': 0, 'Ins': 0, 'Loc': 0}
        self.average_sent_length = 0
        self.average_word_length = 0
        self.average_parts_in_sentence = 0  # how many clauses in sentence on average
        self.average_letter_per_sentence = 0  # average symbols per sentence
        self.percentage_of_unique_words = 0  # number of unique tokens / number of all tokens
        self.lemmas = []  # list of all lemmas in text, includes repetition
        self.bag_of_gramms4 = []  # list of all 4-grams of symbols in text, includes repetition
        self.bag_of_gramms3 = []  # list of all 3-grams of symbols in text, includes repetition
        self.vector = []  # all previous number values combined in one list (used for saving purposes)
        self.data = {}  # dict that will be transfered to json format and saved
        self.sym_chunk = []  # list of syntax chunks
        self.top_words = []  # list of top 100 words in text, ordered
        self.top_verbs = []  # list of top 50 verbs, ordered
        self.phrases = []
        self.process()  # call other class methods to parse text and save json

    def make_symbols(self, size=4):  # fill bag_of_gramms4 and bag_of_gramms3
        for i in range(len(self.text) - size):
            el = self.text[i:i + size]
            if size == 4:
                self.bag_of_gramms4.append(el)
            else:
                self.bag_of_gramms3.append(el)

    def lemmatize_and_fill_params(self):
        sents = list(self.model.process(self.text))
        lemma_text = ''
        total_tokens = 0
        total_word_len = 0  # len in chars
        total_words = 0  # not every token is a word that's why this exists
        words = []
        verbs = []
        self.average_parts_in_sentence = (len(sents) + self.text.count(',') + self.text.count(':')) / len(sents)
        for sent in sents:
            total_tokens += 1

            for word in sent.words:
                if word.upostag != 'punct':
                    if word.lemma not in words:
                        words.append(word.lemma)
                if word.lemma not in self.stop_words or word.form not in self.stop_words:
                    lemma_text += word.lemma + ' '  # lemmatizing text

                try:
                    self.word_parts_dist[word.upostag] += 1
                except KeyError:
                    continue

                if word.upostag == 'VERB':
                    verbs += word.lemma
                if word.upostag == 'NOUN':
                    for case in self.cases_dist.keys():
                        if case in word.feats:
                            self.cases_dist[case] += 1
                            break
                try:
                    self.deprel_dist[word.deprel] += 1
                except KeyError:
                    self.deprel_dist['UNK'] += 1

                if word.upostag != 'PUNCT':
                    total_word_len += len(word.form)
                    total_words += 1
            # this fills chunks from one sentence
            used_tokens = []
            np_sent = {}
            vp_sent = {}
            bad_cycle = 0  # prevents infinite loop
            while len(used_tokens) != len(sent.words) - 2 and bad_cycle < 2:
                for word in sent.words[:-1]:
                    if word.id in used_tokens:
                        continue
                    if word.deprel == 'root':  # suggestion that every verb part starts with root
                        vp_sent[word.id] = word.upostag
                        used_tokens.append(word.id)
                        break
                    elif word.deprel == 'nsubj':  # suggestion that every noun part starts with noun in nom form
                        np_sent[word.id] = word.upostag
                        used_tokens.append(word.id)
                        break
                    if word.head in np_sent.keys():
                        np_sent[word.id] = word.upostag
                        bad_cycle = 0
                        used_tokens.append(word.id)
                        break
                    elif word.head in np_sent.keys():
                        vp_sent[word.id] = word.upostag
                        bad_cycle = 0
                        used_tokens.append(word.id)
                        break
                else:
                    bad_cycle += 1  # if nothing added in np or vp then the cycle considered as bad
            # making 3-grams of syntax chunks
            np_sent = list(OrderedDict(sorted(np_sent.items(), key=lambda t: t[0])).values())
            np_sent.extend(['_', '_'])
            vp_sent = list(OrderedDict(sorted(vp_sent.items(), key=lambda t: t[0])).values())
            vp_sent.extend(['_', '_'])
            for i in range(len(np_sent) - 2):
                self.sym_chunk.append(''.join(np_sent[i:i + 3]))
            for i in range(len(vp_sent) - 2):
                self.sym_chunk.append(''.join(vp_sent[i:i + 3]))
            # end of filling syntax chunks

        wp_dist_sum = sum(self.word_parts_dist.values())
        deprel_sum = sum(self.deprel_dist.values())
        cases_sum = sum(self.cases_dist.values())
        self.average_sent_length = total_words / self.word_parts_dist['<root>']

        # normalize numbers in dicts
        for key in self.word_parts_dist.keys():
            self.word_parts_dist[key] /= wp_dist_sum
        for key in self.deprel_dist.keys():
            self.deprel_dist[key] /= deprel_sum
        for key in self.cases_dist.keys():
            self.cases_dist[key] /= cases_sum
        self.lemmas = lemma_text[:-1].split()
        while ' ' in self.lemmas:
            self.lemmas.remove(' ')
        self.average_word_length = total_word_len / total_words
        self.average_letter_per_sentence = total_word_len / len(sents)
        self.percentage_of_unique_words = len(words) / total_words
        verbs = Counter(verbs)
        self.top_verbs = sorted(verbs, key=lambda x: int(verbs[x]), reverse=True)[:50]

    def make_vector(self):  # filling self.vector
        self.vector.append(self.average_word_length)
        self.vector.append(self.average_sent_length)
        self.vector.append(self.average_parts_in_sentence)
        self.vector.append(self.average_letter_per_sentence)
        self.vector.append(self.average_parts_in_sentence / self.average_letter_per_sentence)
        self.vector.append(self.percentage_of_unique_words)
        self.vector.append(self.word_parts_dist['NOUN'] / self.word_parts_dist['VERB'])
        self.vector.extend(self.phrases)
        self.vector.extend(list(self.word_parts_dist.values()))
        self.vector.extend(list(self.deprel_dist.values()))
        self.vector.extend(list(self.punctuation_dist.values()))
        self.vector.extend(list(self.cases_dist.values()))

    def get_top_words(self):
        counts = Counter(self.lemmas)
        self.top_words = sorted(counts, key=lambda x: int(counts[x]), reverse=True)[1:101]

    def make_phrases(self):
        agr = self.deprel_dist['nmod'] + self.deprel_dist['appos'] + \
              self.deprel_dist['acl'] + self.deprel_dist['det'] \
              + self.deprel_dist['amod'] + self.deprel_dist['nummod'] + self.deprel_dist['compound']
        man = self.deprel_dist['nsubj'] + self.deprel_dist['csubj'] + \
              self.deprel_dist['obj'] + self.deprel_dist['ccomp'] + self.deprel_dist['iobj']
        adj = self.deprel_dist['obl']
        self.phrases.append(agr / self.word_parts_dist['NOUN'])
        self.phrases.append(man / self.word_parts_dist['NOUN'])
        self.phrases.append(adj / self.word_parts_dist['NOUN'])

    def puctuation_distribution(self):
        for key in self.punctuation_dist.keys():
            self.punctuation_dist[key] = self.text.count(key)
        all_punctiation = sum(self.punctuation_dist.values())
        for key in self.punctuation_dist.keys():
            self.punctuation_dist[key] /= all_punctiation

    def save_text_data(self):  # save data to json
        path = self.folder + self.filename + '.json'
        with open(path, 'w', encoding='utf8') as fl:
            json.dump(self.data, fl)

    def process(self):
        self.make_symbols(4)  # makes bag_of 4-gramms
        self.make_symbols(3)  # makes bag_of 3-gramms
        self.lemmatize_and_fill_params()  # process text
        self.make_vector()  # fill vector
        self.make_phrases()
        self.get_top_words()  # get top verbs
        self.data = {'vec': self.vector,
                     'symbols': self.bag_of_gramms4,
                     'symbols_2': self.bag_of_gramms3,
                     'tokens': self.lemmas,
                     'chunk': self.sym_chunk,
                     'top_words': self.top_words,
                     'top_verbs': self.top_verbs}
        self.save_text_data()  # save self.data to json


class CalculatePair:
    # process 2 json files as pair
    def __init__(self, text_1, text_2, folder):
        data_1 = json.load(open(folder + text_1 + '.json', 'r'))
        data_2 = json.load(open(folder + text_2 + '.json', 'r'))
        self.pair_vector = []
        vectorizer = TfidfVectorizer(min_df=0.1)

        texts_length_proportion = len(data_1['tokens']) / len(data_2['tokens'])
        if texts_length_proportion > 1:
            texts_length_proportion = 1 / texts_length_proportion

        token_tf_idf = vectorizer.fit_transform([' '.join(data_1['tokens']), ' '.join(data_2['tokens'])])
        symbol_tf_idf = vectorizer.fit_transform([' '.join(data_1['symbols']), ' '.join(data_2['symbols'])])
        symbol2_tf_idf = vectorizer.fit_transform([' '.join(data_1['symbols_2']), ' '.join(data_2['symbols_2'])])
        chunk_tf_idf = vectorizer.fit_transform([' '.join(data_1['chunk']), ' '.join(data_2['chunk'])])
        top_words_tf_idf = vectorizer.fit_transform([' '.join(data_1['top_words']), ' '.join(data_2['top_words'])])

        self.pair_vector.append(calculate_cos(token_tf_idf[0], token_tf_idf[1]))  # tf-idf
        self.pair_vector.append(calculate_cos(symbol_tf_idf[0], symbol_tf_idf[1]))  # tf-idf for 4-grams
        self.pair_vector.append(calculate_cos(symbol2_tf_idf[0], symbol2_tf_idf[1]))  # tf-idf for 3-grams
        self.pair_vector.append((self.pair_vector[1] + self.pair_vector[2]) / 2)  # mean of grams tf-idf
        self.pair_vector.append(self.pair_vector[0] * self.pair_vector[3])  # token tf-idf * mean grams tf-idf
        self.pair_vector.append(
            self.pair_vector[0] / self.pair_vector[3])  # proportion token tf-idf * mean grams tf-idf
        self.pair_vector.append(calculate_cos(chunk_tf_idf[0], chunk_tf_idf[1]))  # tf-idf of syntax chunks
        self.pair_vector.append(
            calculate_cos(top_words_tf_idf[0], top_words_tf_idf[1]))  # tf-idf top_words (just in case)

        # count mean_dif for every set
        for el in ['tokens', 'symbols', 'symbols_2', 'chunk', 'top_words', 'top_verbs']:
            all_tokens = set(data_1[el] + data_2[el])
            part_token_1 = len(set(data_1[el])) / len(all_tokens)
            part_token_2 = len(set(data_2[el])) / len(all_tokens)
            missing_tokens = (part_token_1 + part_token_2) / 2
            self.pair_vector.append(missing_tokens)

        same_top_words = 0
        for i in range(len(data_1['top_words'])):
            if data_1['top_words'][i] in data_2['top_words']:
                same_top_words += 1
        self.pair_vector.append(same_top_words / 100)
        self.pair_vector.append(texts_length_proportion)

        diff = []
        for i in range(len(data_1['vec'])):
            el_1 = data_1['vec'][i]
            el_2 = data_2['vec'][i]
            el_dif = el_1 - el_2
            if el_dif < 0:
                el_dif *= -1
            if el_1 > el_2:
                el_prop = el_2 / el_1
            else:
                if el_2 == 0:
                    if el_1 == 0:
                        el_prop = 1
                    else:
                        el_prop = 0
                else:
                    el_prop = el_1 / el_2
            diff.append(el_prop)
            diff.append(el_dif)
        self.pair_vector.extend(diff)

        for i in range(len(self.pair_vector)):
            self.pair_vector.append(self.pair_vector[i] ** 2)

    def get_pair_data(self):
        return self.pair_vector


def calculate_cos(tokens_1, tokens_2):
    cos = 1 - spatial.distance.cosine(tokens_1.toarray(), tokens_2.toarray())
    return cos
