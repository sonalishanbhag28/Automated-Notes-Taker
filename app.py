import os
from flask import Flask, render_template, request, redirect, url_for, abort
from werkzeug.utils import secure_filename

import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
# app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'uploads'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = uploaded_file.filename
    if filename != '':
        # file_ext = os.path.splitext(filename)[1]
        # uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))

        # Importing Libraries
        import speech_recognition as sr
        import os
        from pydub import AudioSegment
        from pydub.silence import split_on_silence
        import string
        import re
        from time import sleep
        from tabulate import tabulate
        from repRE import REReplacer
        import requests
        import json
        from nltk.tag import pos_tag
        from textblob import TextBlob
        import pandas as pd
        import nltk
        import subprocess

        nltk.download('vader_lexicon')
        nltk.download('punkt')
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sid = SentimentIntensityAnalyzer()
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Function Definitions
        # Function to convert a list to string

        def listToString(s):
            str1 = " "
            return (str1.join(s))

        # Function for converting speech to text
        r = sr.Recognizer()

        def speechtotext(path):
            var = sr.AudioFile(path)
            with var as source:
                audio = r.record(source)
            return r.recognize_google(audio)

        # Function for preprocessing text
        def preprocessing(name):
            file = open(name, 'r')
            text = file.read()
            file.close()
            rep_word = REReplacer()
            exptext = rep_word.replace(text)
            print("\n\nStep 1. Expand any Contractions\n")
            print(exptext)
            sleep(1)
            print("\nStep 2. Split into words\n")
            # split into words
            from nltk.tokenize import word_tokenize
            tokens = word_tokenize(exptext)
            print(tokens)
            sleep(1)
            # writing preprocessed text into file
            f = open('input.txt', 'w')
            for i in tokens:
                if i == '.':
                    f.write(".\n")
                else:
                    f.write(i+" ")
            f.close()
            print("\nStep 3. Convert to lower case\n")
            # convert to lower case
            tokens = [w.lower() for w in tokens]
            print(tokens)
            sleep(1)
            print("\nStep 4. Remove punctuation from each word\n")
            # remove punctuation from each word
            import string
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            print(stripped[:100])
            sleep(1)
            print("\nStep 5. Remove anything non alphabetic\n")
            # remove remaining tokens that are not alphabetic
            words = [word for word in stripped if word.isalpha()]
            print(words)
            sleep(1)
            print("\nStep 6. Filter out stop words\n")
            # filter out stop words
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if not w in stop_words]
            print(words[:100])
            sleep(1)
            print("\nStep 7. Stem the words\n")
            # stemming of words
            from nltk.stem.porter import PorterStemmer
            porter = PorterStemmer()
            stemmed = [porter.stem(word) for word in words]
            print(stemmed[:100])
            sleep(1)
            print("\nFinal Preprocessed Text:\n")
            return(listToString(stemmed))

        content = speechtotext(filename)
        f = open("converted_text.txt", "w")
        f.write(content)
        f.close()
        print(preprocessing('converted_text.txt'))

        ##########################
        #Start Feature Extraction#
        ##########################

        # features are stored in F1, F2... etc
        # these are arrays which store each senteces value in order of sentences

        print("\n\nStarting Feature Extraction\n")
        # input.txt stores text after preprocessing steps
        filename = r'input.txt'
        with open(filename) as f_obj:
            contents = f_obj.read()

        # splitting into sentences

        def splittosentences(contents):
            senlist = contents.splitlines()
            return senlist

        sentences = splittosentences(contents)

        for sen in sentences:
            sen = sen.rstrip("\n")

        # title processing
        title = sentences[0]
        title = title.rstrip(".")
        titlewords = title.split(" ")

        # commencing feature extraction
        # print(sentences)
        senno = len(sentences)

        def feature1(titlewords, sentences, senno):
            f1 = [0]
            for i in range(0, senno-1):
                f1.append(0)

            for i in range(0, senno):
                sentences[i] = sentences[i].rstrip(".")
                sentences[i] = sentences[i].split(" ")

            for i in range(0, senno):
                for wordtemp in sentences[i]:
                    if wordtemp in titlewords:
                        f1[i] = f1[i]+1
            return f1
        f1 = feature1(titlewords, sentences, senno)

        # UNCOMMENT FOR FEATURE 1 RESULT:
        print("\nFeature 1 Output: ")
        for i in range(0, senno):
            print(f1[i])

        # normalised length calculation

        def feature2(sentences):
            max = 0
            for i in sentences:
                if len(i) > max:
                    max = len(i)

            f2 = [0]
            for i in range(0, senno-1):
                f2.append(0)

            for i in range(0, senno):
                f2[i] = float(len(sentences[i])/max)
            return f2

        f2 = feature2(sentences)

        # FEATURE 2 RESULT
        print("\nFeature 2 Output: ")
        for i in range(0, senno):
            print(f2[i])

        # weighted sentence position scoring

        def feature3(sentences, senno):
            f3 = [0]
            for i in range(0, senno-1):
                f3.append(0)
            for i in range(0, senno):
                f3[i] = (senno-i)/senno
            return f3

        f3 = feature3(sentences, senno)

        # FEATURE 3 RESULT:
        print("\nFeature 3 Output:")
        for i in range(0, senno):
            print(f3[i])

        # numerical data

        def feature4(sentences):
            f4 = [0]
            for i in range(0, senno-1):
                f4.append(0)
            for i in range(0, senno):
                for word in sentences[i]:
                    for char in word:
                        if(char.isdigit()):
                            f4[i] = f4[i]+1
            return f4

        f4 = feature4(sentences)

        # FEATURE 4 RESULT:
        print("\nFeature 4 Output:")
        for i in range(0, senno):
            print(f4[i])

        # FEATURE 5- Proper nouns
        sentences2 = splittosentences(contents)

        def feature5(sentences2, senno):
            f5 = [0]
            for i in range(0, senno):
                f5.append(0)
            # Parsing into parts of speech:
            for i in range(0, senno):
                sentence = sentences2[i]
                tagged_sent = pos_tag(sentence.split())
                propernouns = [word for word,
                               pos in tagged_sent if pos == 'NNP']
                f5[i] = len(propernouns)
            return f5

        f5 = feature5(sentences2, senno)

        f5 = feature5(sentences2, senno)

        # FEATURE 5 RESULT:
        print("\nFeature 5 Output:")
        for i in range(0, senno):
            print(f5[i])

        # creating similarity matrix
        # self comparisons set to 0
        # remove second last line for upper triangular only

        def feature6(sentences, senno):
            simmat = [[0]*senno for x in range(senno)]

            for i in range(0, senno):
                for j in range(i+1, senno):
                    # print(i,j)
                    for word in sentences[j]:
                        if word in sentences[i]:

                            #print("hit at",i,j)
                            simmat[i][j] = simmat[i][j]+1
                            simmat[j][i] = simmat[i][j]
            return simmat

        simmat = feature6(sentences, senno)

        # UNCOMMENT TO PRINT FEATURE 6 RESULT:
        # print("\nFeature 6 (Similarity Matrix)")
        # for i in range(0, senno):
        #     print("\n")
        #     for x in range(0, senno):
        #             print(simmat[i][x], end=" ")

        #print("succesfully completed")

        print("\n")

        # ignore:
        # f1- title words
        # f2- len/max len
        # f3- position scoring, out of 1
        # f4- numerical data
        # f5- proper nouns
        #simmat- matrix

        # DEFUZZIFICATION: process of normalising all the feature vectors
        # and creating the final summary

        # selected sentences for summary will finally be in this list
        sum1 = []

        # defuzzification for similarity matrix
        f6 = []
        for i in range(0, senno):
            f6.append(0)

        for i in range(0, senno):
            m = max(simmat[i])
            m = m/len(sentences[i])
            f6[i] = m

        score = []

        for i in range(0, senno):
            score.append(f3[i]*2+f4[i]*10+f5[i]*5)

        # title sentence always added to summary
        sum1.append(0)

        # adjust the following to increase or lower length of summary
        # you may balance information and conciseness according to requirements

        for i in range(0, senno):
            if score[i] >= 3.0:  # you can change 'cutoff' or 'threshold' value
                if i not in sum1:
                    sum1.append(i)

        # for i in range(0, senno):
            # print(score[i])

        # removing too short sentences according to feature2
        for i in range(0, senno):
            if f2[i] < 0.4:
                if i in sum1 and i != 0:
                    sum1.remove(i)

        # removing repeated or very similar sentences
        # increase number to decrease tolerance for similarity
        for i in range(0, senno):
            if f6[i] > 0.8:
                if i in sum1 and i != 0:
                    sum1.remove(i)

        sum1.sort()

        print("\nSentence Numbers Selected after Feature Extraction: ", end="")
        print(sum1)

        f = open('input.txt', 'r')
        text = f.read()
        f.close()

        sentences = tokenizer.tokenize(text)
        senno = len(sentences)

        sum2 = []

        # We add the additional step of iterating through the list of sentences and calculating and printing polarity scores for each one.
        senti = 0
        score2 = []
        print("\n")
        print("Calculating Sentiment Scores of each Sentence:")
        for sentence in sentences:
            print(sentence)
            scores = sid.polarity_scores(sentence)
            score2.append(abs(sid.polarity_scores(sentence)['compound']))
            senti += sid.polarity_scores(sentence)['compound']
            for key in sorted(scores):
                print('{0}: {1}, '.format(key, scores[key]), end='')
            print()

        print("\n")
        #print("Sentiment score= ")
        # print(senti)

        sum2.append(0)

        for i in range(1, senno-1):
            if abs(sid.polarity_scores(sentences[i])['compound']-sid.polarity_scores(sentences[i+1])['compound']) > 0.1:
                sum2.append(i)

        print("\nSentence Numbers Selected after Sentiment Analysis: ", end="")
        print(sum2)

        count = -1

        # for sentence in sentences:
        #    count+=1
        #    if count in summary:
        #        print(sentence)

        Fsum = []

        for i in range(0, senno):
            if i in sum1 and i in sum2:
                if i not in Fsum:
                    Fsum.append(i)

        temp = []
        for i in range(0, senno):
            if (i in sum1 and i not in sum2) or (i in sum2 and i not in sum1):
                if i not in temp:
                    temp.append(i)

        Fsum.sort()
        temp.sort()

        A = []
        for i in range(0, senno):
            A.append(score[i] - score2[i]*5)

        for i in range(0, senno):
            for j in temp:
                if j == i and A[j] > 0.25:
                    Fsum.append(j)
        Fsum.sort()

        # sort values in temp according to final
        # sentiment and sentence combined score
        # take the higher half

        print("\nSentences in Final Summary: ", end="")
        print(Fsum)

        # FINAL SUMMARY PRINTING:
        print("\nFinal Summary obtained through Hybrid Method\n")
        for i in range(0, senno):
            if i in Fsum:
                print(sentences2[i])

        f = open("summary.txt", "w")
        f.write(listToString(sentences2))
        f.close()

        print("\nSummary successfully generated!")
        print("Summary file saved in summary.txt!\n")

        with open('summary.txt') as f:
            contents = f.read()
            return render_template('index.html', data=contents)

    return redirect(url_for('index'))
