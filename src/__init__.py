import os

import numpy as np
from gensim.models import Word2Vec
from nltk import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

model_file_path = "../lib/word2vec_model.model"


def scan_column(new_training_data):
    # Tokenize the new sentences
    tokenized_new_sentences = [word_tokenize(sentence.lower()) for sentence in new_training_data]

    if os.path.exists(model_file_path):
        # Load your pre-trained Word2Vec model
        existing_model = Word2Vec.load(model_file_path)
    else:
        # Train Word2Vec model
        existing_model = Word2Vec(sentences=tokenized_new_sentences, vector_size=100, window=5, min_count=1, workers=4)

    # Update the existing model with new sentences
    existing_model.build_vocab(tokenized_new_sentences, update=True)
    existing_model.train(tokenized_new_sentences, total_examples=existing_model.corpus_count, epochs=10)

    # Save the updated model
    existing_model.save(model_file_path)


def calculate_distance(array_of_words1, array_of_words2):
    tokenized_array1 = [word_tokenize(word.lower()) for word in array_of_words1]
    tokenized_array2 = [word_tokenize(word.lower()) for word in array_of_words2]

    # Load the model
    model = Word2Vec.load(model_file_path)

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


if __name__ == '__main__':
    # New sentences containing new words
    new_sentences = [
        "grape", "kiwi", "orange",
        "cherry", "pear", "banana",
        "apple", "kiwi", "mango", "apples"
    ]
    scan_column(new_sentences)
    # print("")

    # Example arrays of words
    array1_words = ["apple", "banana", "orange"]
    array2_words = ["apple", "kiwi", "mango"]
    calculate_distance(array1_words, array2_words)
