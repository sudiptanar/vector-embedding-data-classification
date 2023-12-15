from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Load your pre-trained Word2Vec model
existing_model = Word2Vec.load("word2vec_model.model")

# New sentences containing new words
new_sentences = [
    "This is a new sentence with new words.",
    "Another new sentence for training."
]

# Tokenize the new sentences
tokenized_new_sentences = [word_tokenize(sentence.lower()) for sentence in new_sentences]

# Update the existing model with new sentences
existing_model.build_vocab(tokenized_new_sentences, update=True)
existing_model.train(tokenized_new_sentences, total_examples=existing_model.corpus_count, epochs=10)

# Save the updated model
existing_model.save("word2vec_model.model")
