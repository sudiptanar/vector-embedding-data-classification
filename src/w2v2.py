from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np

# Example Word2Vec model (make sure to load your pre-trained model here)
sentences = ["192 171 250 198 552 157 834 10 11 12 780 144 1 167 2 3 4 5 126 205 7 8 9 528 407 41 928",
             "11 12 1 2 3 4 5 6 7 8 9 10",
             "1 2 3 4 5 6 7 8",
             "0 25 15 5 30 20 10",
             "95008 7960 27403 98004 1581 75234 2184 1833 2139 2116 11747 1730 3049 6897 2903 48084 3801",
             "0.92 11.84 15.27 11.88 18.32 14.72 19.85 488.44 48.63 0.09 19.70 353.30 14.19 393.50 12.84 35.81 3.70 13.14 0.23 270.00 17.92 1.50 128.00 0.18 485.00 60.00",
             "QQ NO EG KN CH DJ JR BN ZF IV YI HX SC FZ DX BW QF OE RK VT"]

# Tokenize sentences
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Example array of words
word_array = ["apple", "banana", "orange"]

# Tokenize the array (you may need to preprocess and tokenize based on your specific use case)
tokenized_array = [word_tokenize(word.lower()) for word in word_array]

# Initialize an empty array to store word vectors
word_vectors = []

# Get vectors for each word and append to the word_vectors array
for tokenized_word in tokenized_array:
    vectors_for_word = [model.wv[word] for word in tokenized_word if word in model.wv]
    if vectors_for_word:
        word_vectors.append(np.mean(vectors_for_word, axis=0))
    else:
        # Handle the case where a word is not in the vocabulary
        word_vectors.append(np.zeros(model.vector_size))

# Convert the list of word vectors to a 2D NumPy array
array_vector_representation = np.array(word_vectors)

print("Array of Word Vectors:")
print(array_vector_representation)
