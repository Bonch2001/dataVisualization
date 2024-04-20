
import pandas
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

class DataRecord:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

df = pandas.read_parquet("data/part-00000-ac5c3a2a-c89d-4817-8a00-1ba717b8f279-c000.snappy.parquet", engine = "pyarrow")
# print(data.dtypes)

# print(data)

# np.savetxt(r'np.txt', df.values, fmt='%s', delimiter=' ', encoding='utf-8')

records = [DataRecord(**row.to_dict()) for index, row in df.iterrows()]
""" # records e o lista de obiecte; fiecare obiect e un rand din DataFrame, avand ca atribute coloanele

# print(records[11].legal_names[0])

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Handle numpy arrays, typically coming from data loading issues
    if isinstance(text, np.ndarray):
        # Convert numpy array to list if not empty, else return empty string
        text = text.tolist() if text.size > 0 else ""
    
    # Handle lists (which could come from previous conversion or directly)
    if isinstance(text, list):
        text = ' '.join(map(str, text))  # Ensure conversion to string and concatenate
    
    
    # Check if the input is a string, process if true
    if isinstance(text, str):
        tokens = nltk.word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and not token in stop_words]
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(lemmatized_tokens)
    
    # If none of the above types, raise an error
    raise ValueError(f"Unsupported data type: {type(text)}")

# Apply the function
df['processed_keywords'] = df['keywords'].apply(preprocess_text)

vectorizer = CountVectorizer(min_df=1, max_features=1000)
word_matrix = vectorizer.fit_transform(df['processed_keywords'])

# print("Shape of matrix:", word_matrix.shape) # (36000, 1000)

k = 10  # Set the number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(word_matrix)

# Attach the cluster labels to the original DataFrame
df['cluster'] = clusters

for i in range(k):
    print(f"Cluster {i}:")
    print(df[df['cluster'] == i]['keywords'].head(10), "\n")
 """

df['keywords'] = df['keywords'].astype(str)
# Convert keywords into TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['keywords'])

# Optional: Standardize features by removing the mean and scaling to unit variance
X = StandardScaler(with_mean=False).fit_transform(X)

# HDBSCAN clustering
hdbscan_cluster = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='eom')
clusters = hdbscan_cluster.fit_predict(X)

# Add cluster labels to the DataFrame
df['cluster'] = clusters
print(df)
