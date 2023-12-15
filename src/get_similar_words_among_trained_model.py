from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Example data
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


# model.save("word2vec_model.model")


vector = model.wv['95008']

similar_words = model.wv.most_similar('95008', topn=5)

model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
