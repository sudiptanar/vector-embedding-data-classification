from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Example Word2Vec model (make sure to load your pre-trained model here)
sentences = ["192 171 250 198 552 157 834 10 11 12 780 144 1 167 2 3 4 5 126 205 7 8 9 528 407 41 928 35 Landsdowne Street 1020 ADAMS ST P.O. BOX 16535 24 Frank Lloyd Wright Drive 2300 SOUTH ORCHARD 1000 Seneca Street 1125 Trenton-Harbourton Road 8700 Mason-Montgomery Road No Addresss Line 1004 Middlegate Road 1800 Concord Pike - FOC 2 CE2 - 502 100 Abbott Park Road 102 Route de Noisy 126 Greville Street 12 Electronics Street 880 Caffrey 19 Greenbelt Drive PO Box 79 143 NORTH MONROE ST 11000 Weston Parkway 30 Bearfoot Road 35 West Watkins Mill Road 513 PITTSTOWN ROAD 323 FOSTER STREET Accounts Payable 245 First Street Diamond Bar Gaithersburg Aldershot Anu UPPER MARLBORO Titusville Cambridge Law Courts San Diego Emeryville NEW BRUNSWICK Wilmington Alpharetta Hagerstown Bendminster Greenville St. Petersburg ST CLOUD Bannockburn Welwyn Garden City Bruce Mason Granville Durham DOWNERSGROVE Toronto POUGHKEEPSIE",
             "92122 19044 48207-2392 27701 08560-0200 8010 45040-8006 08906-6500 M3C1L9 15332-1314 11747 08906-6540 19850-5437 7936 72168-0240 10036 10017 19406 15230-0348 3133 3155 2142 2139 200 02451-1420 47933-1958",
             "CAN USA FRA DEU GBR UAS ITA CHN JPN US AUS",
             "DE District of Columbia IL CHES OXON VIC ACT NSW MA NC MD OH ID NJ 96 CA ON",
             "22 57 13 15 39 17 29 120 0 6 31 86 53",
             "20 bags x 4 pieces 12 - 12 oz cans 1 kg pkg. 24 - 12 oz bottles 48 pieces 18 - 500 g pkgs. 12 - 8 oz jars 10 - 200 g glasses 12 - 550 ml bottles 12 - 1 lb pkgs. 12 - 12 oz jars 16 kg pkg. 12 - 200 ml jars 1k pkg. 10 - 500 g pkgs. 10 boxes x 20 bags 36 boxes",
             "Wimmers gute Semmelknödel Chef Anton's Cajun Seasoning Rogede sild Steeleye Stout Queso Cabrales Ipoh Coffee Chai Grandma's Boysenberry Spread Northwoods Cranberry Sauce Chang Ikura Mishi Kobe Niku Chef Anton's Gumbo Mix Gnocchi di nonna Alice Aniseed Syrup Uncle Bob's Organic Dried Pears",
             "0 100 80 70 60 50 40 30 20 10",
             "11 12 57 48 1 2 3 4 5 6 7 8 70 9 41 54 10",
             "38.0000 21.3500 22.0000 21.0500 31.0000 25.0000 97.0000 19.0000 30.0000 10.0000 18.0000 40.0000 13.0000 21.0000",
             "11 12 1 2 3 4 5 6 7 8 9 10",
             "1 2 3 4 5 6 7 8",
             "0 25 15 5 30 20 10",
             "Mellvile New York Providence Braintree Cambridge Westboro Columbia Orlando Georgetow Morristown Bedford Edison Hollis Portsmouth Colorado Springs Boston",
             "95008 7960 27403 98004 1581 75234 2184 1833 2139 2116 11747 1730 3049 6897 2903 48084 3801",
             "1 2 3 4",
             "22 13 14 48 38 16 -40 -25 122 4 5 7 70 41 53 54 32",
             "CHF  (CHF) NOK  (NKr) EUR  (€) USD  (US$) SEK  (SKr) GBX  (£) DKK  (DKr)",
             "1 2 3 4 5 6 7 8"]

# Tokenize sentences
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
# Load your pre-trained Word2Vec model
# model = Word2Vec.load("your_word2vec_model.model")
#
# Example arrays of words
array1_words = ["0 100 80 70 60 50 40 30 20 10"]
array2_words = ["22 57 13 15 39 17 29 120 0 6 31 86 53"]

# Tokenize the arrays
tokenized_array1 = [word_tokenize(word.lower()) for word in array1_words]
tokenized_array2 = [word_tokenize(word.lower()) for word in array2_words]

# Get Word2Vec vectors for each array
# array1_vectors = [model.wv[word] for word in tokenized_array1 if word in model.wv]
# array2_vectors = [model.wv[word] for word in tokenized_array2 if word in model.wv]



# Initialize an empty array to store word vectors
word_vectors1 = []
word_vectors2 = []

# Get vectors for each word and append to the word_vectors array
for tokenized_word in tokenized_array1:
    vectors_for_word = [model.wv[word] for word in tokenized_word if word in model.wv]
    if vectors_for_word:
        word_vectors1.append(np.mean(vectors_for_word, axis=0))
    else:
        # Handle the case where a word is not in the vocabulary
        word_vectors1.append(np.zeros(model.vector_size))

        # Get vectors for each word and append to the word_vectors array
for tokenized_word in tokenized_array2:
    vectors_for_word = [model.wv[word] for word in tokenized_word if word in model.wv]
    if vectors_for_word:
        word_vectors2.append(np.mean(vectors_for_word, axis=0))
    else:
        # Handle the case where a word is not in the vocabulary
        word_vectors2.append(np.zeros(model.vector_size))

# Convert the list of word vectors to a 2D NumPy array
# array_vector_representation = np.array(word_vectors1)

# array1_vectors = word_vectors1
# array2_vectors = word_vectors2



# Calculate cosine similarity
if word_vectors1 and word_vectors2:
    array1_mean_vector = np.mean(word_vectors1, axis=0)
    array2_mean_vector = np.mean(word_vectors2, axis=0)

    cosine_sim = cosine_similarity([array1_mean_vector], [array2_mean_vector])
    distance = 1 - cosine_sim  # Convert similarity to distance (1 - similarity)

    # print("Cosine Similarity:")
    # print(cosine_sim)
    print("Cosine Distance:")
    print(distance)

    # calculate Euclidean distance
    euclidean_distance = np.linalg.norm(array1_mean_vector - array2_mean_vector)
    print("Euclidean Distance:")
    print(euclidean_distance)

    # Calculate Manhattan distance
    manhattan_distance = np.sum(np.abs(array1_mean_vector - array2_mean_vector))
    print("Manhattan Distance:")
    print(manhattan_distance)

    # Calculate Chebyshev distance
    chebyshev_distance = np.max(np.abs(array1_mean_vector - array2_mean_vector))
    print("Chebyshev Distance:")
    print(chebyshev_distance)

else:
    print("No vectors found for one or both arrays.")
