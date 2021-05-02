from django.shortcuts import render, HttpResponse
import glob
import re
from contractions import contractions
import os
import nltk
from nltk.stem import WordNetLemmatizer
import math


# nltk.download('wordnet')
# order of functions
# 1. collectDocuments(path,query) -> Collect the documents to be indexed.
#       1.01 tokenization(document) -> Tokenize the text.
#       here term frequency is also calculated
#   1.1 Index the documents that each term occurs in
# 2. calculatingDocumentFrequency() -> calculating and updating term document
#       with document frequencies
# 3. calculateInverseDocumentFrequency() -> calculating idf with the formula Log10(N/df)
# 4. normalizingTermFrequency() -> term frequency normalization to length normalize the document
# 5. calculate_ntf_idf() -> calculating weights of the documents
# 6. cosine_similarity() -> calculating similarity

euclidean_lengths_for_each_doc = []
final_terms_with_stops = []
list_of_documentID = []
term_document_dictionary = {}
total_documents = 51
cosineSim = []



def index(request):
    global term_document_dictionary
    global list_of_documentID
    global final_terms_with_stops
    global euclidean_lengths_for_each_doc
    global cosineSim
    global total_documents
    euclidean_lengths_for_each_doc = []
    final_terms_with_stops = []
    list_of_documentID = []
    term_document_dictionary = {}
    total_documents = 51
    cosineSim = []
    query = request.POST
    if bool(query):
        print(query)
        collectDocuments('dataset/ShortStories',query['queryholder'])
        calculateDocumentFrequency()
        calculateInverseDocumentFrequency()
        normalizingTermFrequency()
        calculate_ntf_idf()
        cosine_similarity()
        documents_alpha_0 = []
        documents_alpha_0_001 = []
        for item in range(0,50):
            if cosineSim[item] > 0:
                documents_alpha_0.append(item+1)
            if cosineSim[item] + 0.004 > 0.005:
                documents_alpha_0_001.append(item+1)
        print(query['queryholder'])
        print(documents_alpha_0_001)
        print(len(documents_alpha_0_001))
        print(documents_alpha_0)
        print(len(documents_alpha_0))
        context = {
            'list_1' : documents_alpha_0_001,
            'count_1' : len(documents_alpha_0_001),
            'list_2' : documents_alpha_0,
            'count_2' : len(documents_alpha_0),
        }
    else:
        context = {
            'list_1' : "null",
            'count_1' : "null",
            'list_2' : "null",
            'count_2' : "null",
        }
    return render(request, 'index.html', context)


#collecting documents
def collectDocuments(path,input_query):
    print(input_query)
    open(os.path.join(path, str(total_documents) + '.txt'), "w").close()
    f = open(os.path.join(path, str(total_documents) + '.txt'), "w")
    f.write(input_query)
    f.close
    # opening files one by one in the directory
    for filename in glob.glob(os.path.join(path, '*.txt')):
        with open(os.path.join(os.getcwd(), filename), 'r') as f:
            #handling the encoding of the text files
            f = open(filename, "r", encoding="utf8")
            #Reading from file
            file_contents = f.read()
            # print(filename)
            f.close()

            # sending the document for tokenization
            final_tokenized_terms = []
            final_tokenized_terms = tokenization(file_contents)
            final_tokenized_terms.sort()

            #extracting documentid from the path
            path_split = os.path.split(filename)
            document_id = path_split[1][:-4]
            integer_document_id = int(document_id)

            #saving document id
            list_of_documentID.append(integer_document_id)
            # maintaining dictionary with unique terms
            for word in final_tokenized_terms:
                if word in term_document_dictionary.keys():
                    # if key exists then append the document list
                    term_document_dictionary[word][integer_document_id] = 0
                else:
                    # if key does not exist then create a document list for the key
                    term_document_dictionary[word] = [0] * 54
                    term_document_dictionary[word][integer_document_id] = 0

            #calculating term frequencies
            for term in final_terms_with_stops:
                if term in term_document_dictionary.keys():
                    term_document_dictionary[term][integer_document_id] = term_document_dictionary[term][integer_document_id] + 1

#tokenizing
def tokenization(document):
    #CASEFOLDING
    # casefolding or lower casing the whole document
    document_to_work = document.casefold()

    # OPENING CONTRACTIONS
    # making a list for contractions processing
    list_terms = document_to_work.split()
    list_terms_no_contractions = list()
    for word in list_terms:
        if word in contractions:
            # using an imported dictionary for contractions
            list_terms_no_contractions.append(contractions[word])
        else:
            list_terms_no_contractions.append(word)


    # REMOVING PUNCTUATIONS
    document_no_punctuation = ' '.join(list_terms_no_contractions)
    document_no_punctuation = re.sub(r'[^\w\s]', '', document_no_punctuation)

    # REMOVING FINAL LEFT OVER WHITESPACES
    finalised_terms_with_stopwords = document_no_punctuation.split()
    finalised_terms_with_stopwords.sort()

    # print(finalised_terms_with_stopwords)
    global final_terms_with_stops
    final_terms_with_stops = [0]
    # print(len(final_terms_with_stops))

    # lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_with_stops = list()
    for item in finalised_terms_with_stopwords:
        # lemmatized_with_stops.append(lemmatizer.lemmatize(item, 'n'))
        if item != lemmatizer.lemmatize(item, 'v'):
            lemmatized_with_stops.append(lemmatizer.lemmatize(item,'v'))
            # print(item)
        elif item != lemmatizer.lemmatize(item,'r'):
            lemmatized_with_stops.append(lemmatizer.lemmatize(item, 'r'))
            # print(item)
        else:
            lemmatized_with_stops.append(lemmatizer.lemmatize(item))
    final_terms_with_stops = finalised_terms_with_stopwords
    # final_terms_with_stops = finalised_terms_with_stopwords
    # print(len(final_terms_with_stops))

    # REMOVING STOP WORDS AND DUPLICATE WORDS
    # opening stopwords file
    f = open('dataset/Stopword-List.txt', 'r', encoding='utf8')
    stop_words = f.read()
    stop_list = stop_words.split()
    f.close()

    # removing stop words
    finalised_terms_without_stopwords = list(set(finalised_terms_with_stopwords) ^ set(stop_list))
    finalised_terms_without_stopwords = list(finalised_terms_without_stopwords)

    # print(len(finalised_terms_without_stopwords))
    #lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_list = list()
    for item in finalised_terms_without_stopwords:
        # lemmatized_list.append(lemmatizer.lemmatize(item,'n'))
        # print(item)
        if item != lemmatizer.lemmatize(item, 'v'):
            lemmatized_list.append(lemmatizer.lemmatize(item,'v'))
            # print(item)
        elif item != lemmatizer.lemmatize(item,'r'):
            lemmatized_list.append(lemmatizer.lemmatize(item, 'r'))
            # print(item)
        else:
            lemmatized_list.append(lemmatizer.lemmatize(item))
            # print(item)
        # print(lemmatized_list)

    #delete duplicates
    lemmatized_list = list(set(lemmatized_list))
    # print(len(lemmatized_list))
    return finalised_terms_without_stopwords

#calculating document frequency
def calculateDocumentFrequency():
    for term in term_document_dictionary:
        count = total_documents + 1
        listOfDocs = term_document_dictionary[term]
        df = 0
        for item in range(0,count):
            if listOfDocs[item] != 0:
                df = df + 1
        term_document_dictionary[term][count] = df

#calculating inverse document frequency
def calculateInverseDocumentFrequency():
    for term in term_document_dictionary:
        df_location = total_documents + 1
        idf_location = total_documents + 2
        df = term_document_dictionary[term][df_location]
        idf = math.log10(total_documents/df)
        term_document_dictionary[term][idf_location] = idf

#length normalizing term frequencies
def normalizingTermFrequency():
    #0-49
    max_list = list()
    count = total_documents + 1
    document_id = 1
    tfmaxOfDoc = 0
    alpha = 0.0005
    tf = 0
    ntf = 0
    for iter in range(1,count):
        square_for_euclidean_doc = []
        for item in term_document_dictionary:
            square_for_euclidean_doc.append(term_document_dictionary[item][iter] ** 2)
        euc_len = math.sqrt(sum(square_for_euclidean_doc))
        euclidean_lengths_for_each_doc.append(euc_len)

    for term in term_document_dictionary:
        for iter in range(1,count):
            tf = term_document_dictionary[term][iter]
            ntf = float(tf/euclidean_lengths_for_each_doc[iter-1])
            term_document_dictionary[term][iter] = ntf

#ntf-idf for weighting
def calculate_ntf_idf():
    count = total_documents + 1
    idf_location = total_documents + 2
    ntf_idf = 0
    for term in term_document_dictionary:
        for iter in range(1,count):
            ntf = term_document_dictionary[term][iter]
            idf = term_document_dictionary[term][idf_location]
            ntf_idf = ntf * idf
            term_document_dictionary[term][iter] = ntf_idf

#cosine similarity
def cosine_similarity():
    count = total_documents
    query_location = total_documents
    query_ntf_idf = []
    for item in term_document_dictionary:
        query_ntf_idf.append(term_document_dictionary[item][query_location])
    # print(len(query_ntf_idf))
    square_for_euclidean = list()
    for i in query_ntf_idf:
        square_for_euclidean.append(i**2)
    euclidean_length_query = math.sqrt(sum(square_for_euclidean))

    for iter in range(1,count):
        ntf_idf_of_doc = []
        for term in term_document_dictionary:
            ntf_idf_of_doc.append(term_document_dictionary[term][iter])
        ntf_idf_of_doc
        #dot product
        dotProduct = float(0)
        for item in range(0,len(ntf_idf_of_doc)):
             dotProduct = dotProduct + (query_ntf_idf[item]*ntf_idf_of_doc[item])

        square_for_euclidean_doc = []
        for i in ntf_idf_of_doc:
            square_for_euclidean_doc.append(i ** 2)

        euclidean_length_doc = math.sqrt(sum(square_for_euclidean_doc))
        cosSim = dotProduct/(euclidean_length_doc*euclidean_length_query)
        cosineSim.append(cosSim)





