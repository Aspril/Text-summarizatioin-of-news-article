from __future__ import print_function
import theano
import entity2
import rbm
import re
from nltk.corpus import stopwords
import nltk
import collections
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import math
from operator import itemgetter
import pandas as pd
import sys
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
import os


class RBM_summarizer():
    porter = PorterStemmer()
    lemmer = WordNetLemmatizer()
    stemmer = nltk.stem.porter.PorterStemmer()
    WORD = re.compile(r'\w+')
    caps = "([A-Z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"
    stop = set(stopwords.words('english'))

    def __init__(self, content):
        self.content = content

    @staticmethod
    def split_into_sentences(text):
        "regex pattern to find sentences within quotes"
        text = " " + text + "  "
        text = text.replace("\n", " ")
        text = text.replace('..."', '"')
        text = text.replace('... ', ',')
        text = re.sub(RBM_summarizer.prefixes, "\\1<prd>", text)
        text = re.sub(RBM_summarizer.websites, "<prd>\\1", text)
        if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
        text = re.sub("\s" + RBM_summarizer.caps + "[.] ", " \\1<prd> ", text)
        text = re.sub(RBM_summarizer.acronyms + " " + RBM_summarizer.starters, "\\1<stop> \\2", text)
        text = re.sub(RBM_summarizer.caps + "[.]" + RBM_summarizer.caps + "[.]" + RBM_summarizer.caps + "[.]",
                      "\\1<prd>\\2<prd>\\3<prd>", text)
        text = re.sub(RBM_summarizer.caps + "[.]" + RBM_summarizer.caps + "[.]", "\\1<prd>\\2<prd>", text)
        text = re.sub(" " + RBM_summarizer.suffixes + "[.] " + RBM_summarizer.starters, " \\1<stop> \\2", text)
        text = re.sub(" " + RBM_summarizer.suffixes + "[.]", " \\1<prd>", text)
        text = re.sub(" " + RBM_summarizer.caps + "[.]", " \\1<prd>", text)
        text = re.sub("'s|’s|'", '', text)
        text = re.sub("—", " ", text)
        text = re.sub("-", " ", text)
        if "”" in text: text = text.replace(".”", "”.")
        if "\"" in text: text = text.replace(".\"", "\".")
        if "!" in text: text = text.replace("!\"", "\"!")
        if "?" in text: text = text.replace("?\"", "\"?")
        # #if "," in text: text = text.replace(",\"","\",")

        text = text.replace(".", ".<stop>")
        text = text.replace("?", "?<stop>")
        text = text.replace("!", "!<stop>")
        # text = text.replace(",","!<stop>")
        text = text.replace("<prd>", ".")
        text = text.replace("    ", " ")

        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]
        for sen in sentences:
            sen = re.sub("'s|’s|'", '', sen)
        return sentences

    @staticmethod
    def thematicFeature(tokenized_sentences):
        word_list = []
        for sentence in tokenized_sentences:
            for word in sentence:
                try:
                    word = ''.join(e for e in word if e.isalnum())
                    word_list.append(word)
                except Exception as e:
                    print("ERR")
        counts = Counter(word_list)
        number_of_words = len(counts)
        most_common = counts.most_common(20)
        thematic_words = []
        for data in most_common:
            thematic_words.append(data[0])
        scores = []
        for sentence in tokenized_sentences:
            score = 0
            for word in sentence:
                try:
                    word = ''.join(e for e in word if e.isalnum())
                    if word in thematic_words:
                        score = score + 1
                except Exception as e:
                    print("ERR")
            score = 1.0 * score / (number_of_words)
            scores.append(score)
        return scores

    @staticmethod
    def remove_stop_words(sentences):
        tokenized_sentences = []
        for sentence in sentences:
            tokens = []
            split = sentence.lower().split()
            for word in split:
                if word not in RBM_summarizer.stop:
                    try:

                        tokens.append(RBM_summarizer.lemmer.lemmatize(word))
                    except:
                        tokens.append(word)
            tokenized_sentences.append(tokens)
        return (tokenized_sentences)

    @staticmethod
    def posTagger(tokenized_sentences):
        tagged = []
        for sentence in tokenized_sentences:
            tag = nltk.pos_tag(sentence, tagset="universal")
            tagged.append(tag)
        return tagged

    @staticmethod
    def sentenceLength(tokenized_sentences):
        count = []
        maxLength = sys.maxsize
        for sentence in tokenized_sentences:
            num_words = 0
            for word in sentence:
                num_words += 1
            if num_words < 3:
                count.append(0)
            else:
                count.append(num_words)

        count = [1.0 * x / (maxLength) for x in count]
        return count

    @staticmethod
    def upperCaseFeature(sentences):
        tokenized_sentences2 = RBM_summarizer.remove_stop_words_without_lower(sentences)
        upper_case = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        scores = []
        for sentence in tokenized_sentences2:
            score = 0
            for word in sentence:
                if word[0] in upper_case:
                    score = score + 1
            scores.append(1.0 * score / len(sentence))
        return scores

    @staticmethod
    def remove_stop_words_without_lower(sentences):
        tokenized_sentences = []
        for sentence in sentences:
            tokens = []
            split = sentence.split()
            for word in split:
                if word.lower() not in RBM_summarizer.stop:
                    try:

                        tokens.append(word)
                    except:
                        tokens.append(word)

            tokenized_sentences.append(tokens)
        return tokenized_sentences

    @staticmethod
    def text_to_vector(text):

        stop = set(stopwords.words('english'))
        words = RBM_summarizer.WORD.findall(text)
        return collections.Counter(words)

    @staticmethod
    def get_cosine(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
        sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    @staticmethod
    def centroidSimilarity(sentences, tfIsfScore):
        centroidIndex = tfIsfScore.index(max(tfIsfScore))
        scores = []
        for sentence in sentences:
            vec1 = RBM_summarizer.text_to_vector(sentences[centroidIndex])
            vec2 = RBM_summarizer.text_to_vector(sentence)

            score = RBM_summarizer.get_cosine(vec1, vec2)
            scores.append(score)
        return scores

    @staticmethod
    def similarityScores(tokenized_sentences):
        scores = []
        for sentence in tokenized_sentences:
            score = 0;
            for sen in tokenized_sentences:
                if sen != sentence:
                    score += RBM_summarizer.similar(sentence, sen)
            scores.append(score)
        return scores

    @staticmethod
    def tfIsf(tokenized_sentences):
        scores = []
        COUNTS = []
        for sentence in tokenized_sentences:
            counts = collections.Counter(sentence)
            isf = []
            score = 0
            for word in counts.keys():
                count_word = 1
                for sen in tokenized_sentences:
                    for w in sen:
                        if word == w:
                            count_word += 1
                score = score + counts[word] * math.log(count_word - 1)
            scores.append(score / len(sentence))
        return scores

    @staticmethod
    def similar(tokens_a, tokens_b):
        # Using Jaccard similarity to calculate if two sentences are similar
        ratio = len(set(tokens_a).intersection(tokens_b)) / float(len(set(tokens_a).union(tokens_b)))
        return ratio

    @staticmethod
    def sentencePosition(paragraphs):
        scores = []
        for para in paragraphs:
            sentences = RBM_summarizer.split_into_sentences(para)
            if len(sentences) == 1:
                scores.append(1.0)
            elif len(sentences) == 2:
                scores.append(1.0)
                scores.append(1.0)
            else:
                scores.append(1.0)
                for x in range(len(sentences) - 2):
                    scores.append(0.0)
                scores.append(1.0)
        return scores

    @staticmethod
    def properNounScores(tagged):
        scores = []
        for i in range(len(tagged)):
            score = 0
            for j in range(len(tagged[i])):
                if (tagged[i][j][1] == 'NOUN' or tagged[i][j][1] == 'NNP' or tagged[i][j][1] == 'NNPS' or tagged[i][j][
                    1] == 'NN'):
                    score += 1
            scores.append(score / float(len(tagged[i])))
        return scores

    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def numericToken(tokenized_sentences):
        scores = []
        for sentence in tokenized_sentences:
            score = 0
            for word in sentence:
                if RBM_summarizer.is_number(word):
                    score += 1
            scores.append(score / float(len(sentence)))
        return scores

    @staticmethod
    def namedEntityRecog(sentences):
        counts = []
        for sentence in sentences:
            count = entity2.ner(sentence)
            counts.append(count)
        return counts

    def get_summary(self):

        paragraphs = self.content.split('\n\n')
        sentences = RBM_summarizer.split_into_sentences(self.content)
        tokenized_sentences = RBM_summarizer.remove_stop_words(sentences)
        uppercase_score = RBM_summarizer.upperCaseFeature(sentences)
        namedEntityRecogScore = RBM_summarizer.namedEntityRecog(sentences)
        sentencePosScore = RBM_summarizer.sentencePosition(sentences)
        sentenceParaScore = RBM_summarizer.sentencePosition(paragraphs)
        thematicFeatureScore = RBM_summarizer.thematicFeature(tokenized_sentences)
        tagged = RBM_summarizer.posTagger(tokenized_sentences)
        tfIsfScore = RBM_summarizer.tfIsf(tokenized_sentences)
        similarityScore = RBM_summarizer.similarityScores(tokenized_sentences)
        numericTokenScore = RBM_summarizer.numericToken(tokenized_sentences)
        sentenceLengthScore = RBM_summarizer.sentenceLength(tokenized_sentences)
        properNounScore = RBM_summarizer.properNounScores(tagged)
        centroidSimilarityScore = RBM_summarizer.centroidSimilarity(sentences, tfIsfScore)

        featureMat = np.zeros((len(sentences), 8))

        featureMatrix = []
        featureMatrix.append(thematicFeatureScore)
        featureMatrix.append(sentencePosScore)
        featureMatrix.append(sentenceLengthScore)
        featureMatrix.append(sentenceParaScore)
        featureMatrix.append(properNounScore)
        featureMatrix.append(numericTokenScore)
        featureMatrix.append(namedEntityRecogScore)
        featureMatrix.append(tfIsfScore)
        featureMatrix.append(centroidSimilarityScore)

        for i in range(8):
            for j in range(len(sentences)):
                featureMat[j][i] = featureMatrix[i][j]

        feature_sum = []

        for i in range(len(np.sum(featureMat, axis=1))):
            feature_sum.append(np.sum(featureMat, axis=1)[i])

        temp = rbm.test_rbm(dataset=featureMat, learning_rate=0.1, training_epochs=14, batch_size=5, n_chains=5,
                            n_hidden=8)

        enhanced_feature_sum = []
        enhanced_feature_sum2 = []

        for i in range(len(np.sum(temp, axis=1))):
            enhanced_feature_sum.append([np.sum(temp, axis=1)[i], i])
            enhanced_feature_sum2.append(np.sum(temp, axis=1)[i])

        enhanced_feature_sum.sort(key=lambda x: x[0])

        length_to_be_extracted = len(enhanced_feature_sum) / 4

        extracted_sentences = []
        extracted_sentences.append([sentences[0], 0])

        indeces_extracted = []
        indeces_extracted.append(0)

        for x in range(int(length_to_be_extracted)):
            if (enhanced_feature_sum[x][1] != 0):
                extracted_sentences.append([sentences[enhanced_feature_sum[x][1]], enhanced_feature_sum[x][1]])
                indeces_extracted.append(enhanced_feature_sum[x][1])

        extracted_sentences.sort(key=lambda x: x[1])

        finalText = ""

        for i in range(len(extracted_sentences)):
            finalText = finalText + extracted_sentences[i][0]

        return finalText

