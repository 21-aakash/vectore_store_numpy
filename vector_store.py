import numpy as np

class VectorStore:
    def __init__(self):
        self.vector_data={}  #dictionary to store vectors
        self.vector_index={} # dictionary to store index of vector for retrieval 
    
    def add_vector(self, vector_id, vector):

        """
        Add a vector to the store
        args: 
        vector_id: string/int unique-id of the vector
        vector: numpy array representing the vector
        
        """


        self.vector_data[vector_id]=vector
        self.update_index(vector_id, vector)
 
    
    def get_vector(self, vector_id):
        """
        Get a vector from the store
        args: 
        vector_id: string/int unique-id of the vector
        returns: numpy array representing the vector
        
        """
        return self.vector_data.get(vector_id)
    

    def update_index(self, vector_id, vector):

        """
        Update the indexing structure  of the vector store 
        args: 
        vector_id: string/int unique-id of the vector
        vector: numpy array representing the vector
        
        """
        for existing_id , existing_vector in self.vector_data.items():
            similarity = np.dot(vector, existing_vector)/(np.linalg.norm(vector)*np.linalg.norm(existing_vector))
            if existing_id not in self.vector_index:
                self.vector_index[existing_id]={}
            self.vector_index[existing_id][vector_id]=similarity #This stores the similarity score between the new vector (vector_id) and an already stored vector (existing_id).

# This line stores the cosine similarity score between two vectors (existing_id and vector_id) in a dictionary-based data structure       

# self.vector_index = {
#     "A": {"C": 0.99},
#     "B": {"C": 0.98}
# }
     
    def find_similar_vectors(self, query_vector, num_results):
        """
        Find the most similar vectors to a query vector
        args: 
        query_vector: numpy array representing the query vector
        num_results: int number of similar vectors to return
        returns: list of tuples (vector_id, similarity_score) 
        """
        similarity_scores = []
        for vector_id, vector in self.vector_data.items():
            similarity = np.dot(query_vector, vector)/(np.linalg.norm(query_vector)*np.linalg.norm(vector))
            similarity_scores.append((vector_id, similarity))
            #sorting in descending order
            #For the tuple ("vector1", 0.85), x[1] is 0.85.
            similarity_scores.sort( key=lambda x: x[1], reverse=True)
            #returmn top num_results
        return similarity_scores[:num_results]
    
    
    

