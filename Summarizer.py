class Summarizer():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    def __init__(self, news_content):
        self.content = news_content

    @staticmethod
    def text_to_clean_sents(content):
        import re
        import json
        from nltk.tokenize import sent_tokenize

        news_content = content.replace('\n', ' ')
        news_content = re.sub(r'(\.”)|(\.(?=([a-zA-Z]{3,})))|(\.“)', r'. ', news_content)
        news_content = re.sub(r'\.\s(?=([0-9,]+))', r'.', news_content)

        reg_pattern = re.compile('[{}]'.format(re.escape('?|*|@|(|)|~|“|"|”|,|:')))

        sentence_tokens = sent_tokenize(news_content)

        processed_sents = [re.sub(reg_pattern, '', sent) for sent in sentence_tokens]

        with open('word_contractions.json') as f_in:
            word_contractions = json.load(f_in)

        processed_sents2 = []

        for sen in processed_sents:
            for i in word_contractions:
                sen = re.sub(i, word_contractions[i], sen)
            sen = re.sub("'s|’s|'", '', sen)
            sen = re.sub("—", " ", sen)
            sen = re.sub("-", " ", sen)
            processed_sents2.append(sen)

        return (processed_sents2)

    def word_counts_based_summary(self):
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        from nltk.tag import pos_tag
        from nltk.corpus import wordnet
        from nltk.corpus import stopwords

        processed_sents2 = Summarizer.text_to_clean_sents(self.content)

        stp = set(stopwords.words('english'))

        final_tokens = {}

        sents_noun_score = []
        sents_numeric_score = []

        for sent in processed_sents2:
            sen_noun_score = 0
            sen_numeric_score = 0

            word_tokens = [w.lower() for w in word_tokenize(sent) if w.lower() not in stp and len(w) > 2]

            tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}

            for word in word_tokens:

                if word.isdigit():
                    sen_numeric_score += 1

                tag = tag_dict.get(pos_tag([word])[0][1][0].upper(), wordnet.NOUN)

                if tag == 'n':
                    sen_noun_score += 1

                word_lemma = WordNetLemmatizer().lemmatize(word, pos=tag)

                if word_lemma in final_tokens:
                    final_tokens[word_lemma] += 1
                else:
                    final_tokens[word_lemma] = 1

            sents_noun_score.append(sen_noun_score)
            sents_numeric_score.append(sen_numeric_score)

        final_tokens = {token: freq for token, freq in final_tokens.items() if freq >= 2}

        sents_freq_score = []

        final_scores = []

        for sent in processed_sents2:
            word_tokens = [w.lower() for w in word_tokenize(sent) if w.lower() not in stp and len(w) > 2]
            tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
            score = 0
            for word in word_tokens:
                tag = tag_dict.get(pos_tag([word])[0][1][0].upper(), wordnet.NOUN)
                word_lemma = WordNetLemmatizer().lemmatize(word, pos=tag)
                score = score + final_tokens.get(word, 0)
            sents_freq_score.append(score)

        sents_freq_score = list(
            map(lambda x: (x - min(sents_freq_score)) / (max(sents_freq_score) - min(sents_freq_score)),
                sents_freq_score))
        sents_noun_score = list(
            map(lambda x: (x - min(sents_noun_score)) / (max(sents_noun_score) - min(sents_noun_score)),
                sents_noun_score))
        sents_numeric_score = list(
            map(lambda x: ((x - min(sents_numeric_score)) / (max(sents_numeric_score) - min(sents_numeric_score)))
                if min(sents_numeric_score) != max(sents_numeric_score) else 1
                ,
                sents_numeric_score))


        for i in range(len(processed_sents2)):
            final_score = sents_freq_score[i] + sents_noun_score[i] + sents_numeric_score[i]
            final_scores.append((i, final_score))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        no_of_sents_summ = len(processed_sents2) // 4
        summ = final_scores[:no_of_sents_summ]
        summ.sort(key=lambda x: x[0])
        summary = ""
        for x, y in summ:
            summary += processed_sents2[x]

        return summary


    @staticmethod
    def load_embeddings():
        import numpy as np
        word_embeddings = {}
        f = open('glove.6B.100d.txt', encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = coefs
        f.close()
        return (word_embeddings)

    @staticmethod
    def get_similarity_matrix(processed_sents):
        import numpy as np
        sentence_vectors = []
        word_embeddings = Summarizer.load_embeddings()

        for i in processed_sents:
            if len(i) != 0:
                v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 0.001)
            else:
                v = np.zeros((100,))
            sentence_vectors.append(v)

        similarity_matrix = np.zeros((len(sentence_vectors), len(sentence_vectors)))

        from sklearn.metrics.pairwise import cosine_similarity

        for i in range(len(sentence_vectors)):
            for j in range(len(sentence_vectors)):
                if i != j:
                    similarity_matrix[i][j] = \
                    cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[0, 0]

        return similarity_matrix

    def pagerank_summarizer(self):
        import re
        news_content = self.content.replace('\n', ' ')
        news_content = re.sub(r'(\.”)|(\.(?=([a-zA-Z]{3,})))|(\.“)', r'. ', news_content)
        news_content = re.sub(r'\.\s(?=([0-9,]+))', r'.', news_content)

        from nltk.tokenize import sent_tokenize

        all_sentences = sent_tokenize(news_content)
        all_sentences = [sent[:-1] for sent in all_sentences]

        sentences = Summarizer.text_to_clean_sents(self.content)
        sim_matrix = Summarizer.get_similarity_matrix(sentences)

        all_sentences = sent_tokenize(self.content)

        import networkx as nx

        nx_graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(nx_graph)

        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(all_sentences)), reverse=True)

        summary = ''

        no_of_sents_summ = len(all_sentences) // 4

        for i in range(no_of_sents_summ):
            summary += ranked_sentences[i][1]

        return summary

    def get_summary_using_dependency_parsing(self):
        import spacy
        from nltk.tokenize import sent_tokenize, word_tokenize
        nlp = spacy.load('en_core_web_sm')

        res = ' '.join(Summarizer.text_to_clean_sents(self.content))

        all_sents = sent_tokenize(res)
        sent_scores = []

        for i in range(len(all_sents)):
            sent_score = 0
            doc = nlp(all_sents[i])
            sent_score = sent_score + len(doc.ents)

            if len(doc.ents) > 0:
                for token in doc:
                    if len(list(token.children)) > 0:
                        for child in token.children:
                            if str(child) in [str(i) for i in list(doc.ents)]:
                                sent_score = sent_score + len(doc.ents)
                                break

            #         for imp_wrd in get_imp_words(content):
            #             if imp_wrd in word_tokenize(all_sents[i]):
            #                 sent_score += sent_score + 0.1

            sent_scores.append((i, sent_score / len(all_sents[i])))

        sent_scores.sort(key=lambda x: x[1], reverse=True)
        no_of_sents_summ =len(all_sents) // 4
        summary_ids = sent_scores[:no_of_sents_summ]
        summary_ids.sort(key=lambda x: x[0])
        summary = ""
        for i, j in summary_ids:
            summary = summary + all_sents[i]

        return summary
