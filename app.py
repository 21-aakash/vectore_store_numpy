from vector_store import VectorStore
import numpy as np

vector_store = VectorStore()

sentences=["I love machine learning", "I love deep learning", "I love natural language processing"]

#tokenisation and vocabulary building 

vocabulary = set()


for sentence in sentences:
    tokens = sentence.lower().split()
    vocabulary.update(tokens)

#assign unique integer id to each word in the vocabulary
#This line creates a dictionary that maps each word in a given vocabulary to a unique index.


word_to_index={word: i for i, word in enumerate(vocabulary)}

#vectorisation

# converts sentences into numerical vectors using a bag-of-words (BoW) approach. It creates a vector representation where each word's count is stored in a fixed-size array.

sentence_vectors = {}

for sentence in sentences:
    vector = np.zeros(len(vocabulary))  # Step 1: Initialize a zero vector
    tokens = sentence.lower().split()   # Step 2: Tokenize the sentence
    for token in tokens:
        vector[word_to_index[token]] += 1  # Step 3: Increment count in vector
    
    sentence_vectors[sentence] = vector  # Step 4: Store vector representation


#add vector to vector store

for sentence, vector in sentence_vectors.items():
    vector_store.add_vector(sentence, vector)


query_sentence = "I love deep learning"

query_vector = np.zeros(len(vocabulary))

query_tokens = query_sentence.lower().split()

for token in query_tokens:
    query_vector[word_to_index[token]] += 1


similar_sentences = vector_store.find_similar_vectors(query_vector, num_results=2)

print("query sentence ", query_sentence)
print("similar sentences ")

for sentence, similarity in similar_sentences:
    print(f"{sentence} : similarity={similarity: .4f}")
