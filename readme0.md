output : 


query sentence  I love deep learning
similar sentences 
I love deep learning : similarity= 1.0000
I love machine learning : similarity= 0.7500


---

## **Understanding the VectorStore Class**

The `VectorStore` class provides a simple way to store, index, and retrieve vectors. It is designed to store numerical vectors, compute their similarities using **cosine similarity**, and find the most relevant vectors given a query.

---

### **1. Class Initialization (`__init__` method)**  

```python
class VectorStore:
    def __init__(self, dimension):
        self.vector_data={}  # Dictionary to store vectors
        self.vector_index={} # Dictionary to store the index of vector similarities for retrieval 
```

- The constructor initializes two dictionaries:
  - `vector_data`: Stores vectors with their unique IDs.
  - `vector_index`: Maintains a similarity index between stored vectors.

---

### **2. Adding a Vector (`add_vector` method)**  

```python
def add_vector(self, vector_id, vector):
    """
    Add a vector to the store
    args: 
    vector_id: string/int unique-id of the vector
    vector: numpy array representing the vector
    """
    self.vector_data[vector_id]=vector
    self.update_index(vector_id, vector)
```

- **Stores** the provided `vector` using `vector_id` as a key.  
- **Calls `update_index`** to compute similarities with other stored vectors.  

**Example:**  
If we add `"vector1"` with values `[1, 0, 1]`, it is stored as:  
```python
self.vector_data = {"vector1": np.array([1, 0, 1])}
```

---

### **3. Retrieving a Vector (`get_vector` method)**  

```python
def get_vector(self, vector_id):
    """
    Get a vector from the store
    args: 
    vector_id: string/int unique-id of the vector
    returns: numpy array representing the vector
    """
    return self.vector_data.get(vector_id)
```

- Looks up a stored vector using its ID.
- Returns `None` if the ID does not exist.

---

### **4. Updating the Index (`update_index` method)**  

```python
def update_index(self, vector_id, vector):
    """
    Update the indexing structure of the vector store 
    args: 
    vector_id: string/int unique-id of the vector
    vector: numpy array representing the vector
    """
    for existing_id, existing_vector in self.vector_data.items():
        similarity = np.dot(vector, existing_vector) / (np.linalg.norm(vector) * np.linalg.norm(existing_vector))
        if existing_id not in self.vector_index:
            self.vector_index[existing_id] = {}
        self.vector_index[existing_id][vector_id] = similarity
```

- Computes **cosine similarity** between the new vector and every previously stored vector.  
- Stores similarity scores in `self.vector_index`.

📌 **Cosine Similarity Formula:**  
\[
\text{similarity} = \frac{A \cdot B}{||A|| \times ||B||}
\]
Where:
- \( A \) and \( B \) are vectors.
- \( ||A|| \) and \( ||B|| \) are their magnitudes.

**Example:**  
If `"vector1"` = `[1, 0, 1]` and `"vector2"` = `[0, 1, 1]`, the cosine similarity is computed and stored.

📝 **Structure of `self.vector_index`:**
```python
{
    "vector1": {"vector2": 0.707},  # Similarity between vector1 and vector2
    "vector2": {"vector1": 0.707}
}
```

---

### **5. Finding Similar Vectors (`find_similar_vectors` method)**  

```python
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
        similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
        similarity_scores.append((vector_id, similarity))
    similarity_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity score
    return similarity_scores[:num_results]
```

📌 **Steps:**
1. **Computes cosine similarity** between `query_vector` and each stored vector.
2. **Stores results** in `similarity_scores` as `(vector_id, similarity_score)`.
3. **Sorts** in descending order.
4. **Returns the top `num_results` vectors**.

🔹 **Example Usage:**
If we query `"vector3"`, the function returns the two most similar stored vectors:
```python
[
    ("vector1", 0.85), 
    ("vector2", 0.75)
]
```

---

## **Summary**
| **Method**          | **Purpose** |
|---------------------|------------|
| `__init__()`       | Initializes vector storage and similarity index |
| `add_vector()`     | Stores a vector and updates the similarity index |
| `get_vector()`     | Retrieves a vector by ID |
| `update_index()`   | Updates similarity relationships between stored vectors |
| `find_similar_vectors()` | Finds the most similar stored vectors to a given query |

---

