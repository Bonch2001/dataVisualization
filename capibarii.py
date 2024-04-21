from asyncio.windows_events import NULL
from logging.handlers import RotatingFileHandler
from tkinter.filedialog import SaveFileDialog
import matplotlib
from matplotlib import pyplot as plt
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

def to_list(entry):
    if isinstance(entry, np.ndarray):
        return entry.tolist()
    return entry

df = pandas.read_parquet("data/part-00000-ac5c3a2a-c89d-4817-8a00-1ba717b8f279-c000.snappy.parquet", engine = "pyarrow")
df = df.applymap(to_list)

# print(df.dtypes)
# np.savetxt(r'np.txt', df.values, fmt='%s', delimiter=' ', encoding='utf-8')

records = [DataRecord(**row.to_dict()) for index, row in df.iterrows()]
# records e o lista de obiecte; fiecare obiect e un rand din DataFrame, avand ca atribute coloanele
# print(records[11].legal_names[0])
# print("###", records[3].twitter, "###")

def normalize_to_100(values):
    total = sum(values)
    return [(x / total) * 100 for x in values]

def count_matching_records(records, equal_conditions=None, non_empty_conditions=None, switch=0):
    if equal_conditions is None:
        equal_conditions = {}
    if non_empty_conditions is None:
        non_empty_conditions = []
    count = 0
    total = 0
    if switch==0:
        for record in records:
            if all(getattr(record, attr) == value for attr, value in equal_conditions.items()):
                total += 1
                if all(getattr(record, attr) not in [None, ''] for attr in non_empty_conditions):
                    count += 1
        return count/total
    else:
        for record in records:
            if all(getattr(record, attr) == value for attr, value in equal_conditions.items()):
                if all(getattr(record, attr) not in [None, ''] for attr in non_empty_conditions):
                    count += 1
        return count


# domeniu vs. social media
sustainability = ['Engineering & Construction Services', "Professional Associations", "Health Care Delivery", "Agricultural Products", "Meat, Poultry & Dairy", "Food Retailers & Distributors"]
social_media_sites = ['facebook', 'twitter', 'instagram', 'linkedin', 'youtube']
results = []
for s in sustainability:
    result = count_matching_records(records, equal_conditions={'sustainable_industry': s}, non_empty_conditions=social_media_sites)
    results.append(result*100)
plt.figure(figsize=(10,5))
plt.bar(sustainability, results)
plt.title('Share of social media vs. total number of companies/category')
plt.xticks(rotation=45, ha = 'right', fontsize=10)
plt.tight_layout()  
plt.show()


# % din cele prezente pe toate social medias pentru fiecare domeniu
sustainability = ['Engineering & Construction Services', "Professional Associations", "Health Care Delivery", "Agricultural Products", "Meat, Poultry & Dairy", "Food Retailers & Distributors"]
social_media_sites = ['facebook', 'twitter', 'instagram', 'linkedin', 'youtube']
pies = []
for s in sustainability:
    result = count_matching_records(records, equal_conditions={'sustainable_industry': s}, non_empty_conditions=social_media_sites, switch=1)
    pies.append(result)
pies = normalize_to_100(pies)
plt.figure(figsize=(6, 6))
plt.pie(pies, labels=sustainability, autopct='%1.1f%%', startangle=90, colors=['gold', 'yellowgreen', 'lightcoral', 'skyblue', 'red', 'darkgreen'])
plt.axis('equal')
plt.title('Share of social media in all categories')
plt.show()


# social media vs year of founding
social_media_per_ages = [0, 0, 0, 0]
for i in range(len(records)):
    if records[i].facebook != None and records[i].twitter != None and records[i].instagram != None and records[i].linkedin != None and records[i].youtube != None:
        if records[i].year_founded != None:
            if int(records[i].year_founded) < 1900:
                social_media_per_ages[0] += 1
            elif int(records[i].year_founded) < 1960:
                social_media_per_ages[1] += 1
            elif int(records[i].year_founded) < 2000:
                social_media_per_ages[2] += 1
            else:
                social_media_per_ages[3] += 1
social_media_per_ages = normalize_to_100(social_media_per_ages)
plt.figure(figsize=(6, 6))
plt.pie(social_media_per_ages, labels=["<1899","1900-1959","1960-1999","2000-2024"], autopct='%1.1f%%', startangle=90, colors=['gold', 'yellowgreen', 'lightcoral', 'lightblue'])
plt.axis('equal')
plt.title('Share of social media vs. year of founding')
plt.show()


# social media vs employee count
social_media_per_size = [0, 0, 0, 0, 0]
for i in range(len(records)):
    if records[i].facebook != None and records[i].twitter != None and records[i].instagram != None and records[i].linkedin != None and records[i].youtube != None:
        if records[i].employee_count != None:
            if float(records[i].employee_count) < 10:
                social_media_per_size[0] += 1
            elif float(records[i].employee_count) < 100:
                social_media_per_size[1] += 1
            elif float(records[i].employee_count) < 1000:
                social_media_per_size[2] += 1
            elif float(records[i].employee_count) < 10000:
                social_media_per_size[3] += 1
            else:
                social_media_per_size[4] += 1
fig, ax = plt.subplots()
ax.pie(social_media_per_size, labels=["1-9","10-99","100-999","1000-9999", "10000+"], autopct='%1.1f%%', startangle=90, explode=(0, 0, 0, 0.1, 0.2), colors=['gold', 'red', 'blue', 'pink', 'darkgreen'])
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)
ax.axis('equal')  
plt.title('Share of social media vs. employee number')
plt.show()


# social media vs. bussines model
models = ['Manufacturing', 'Services', 'Non Profit', 'Wholesale', 'Retail', 'Education', 'Government']
social_media_sites = ['facebook', 'twitter', 'instagram', 'linkedin', 'youtube']
social_media_per_model = [0, 0, 0, 0, 0, 0, 0]
for i in range(len(records)):
    if records[i].facebook != None and records[i].twitter != None and records[i].instagram != None and records[i].linkedin != None and records[i].youtube != None:
        if records[i].business_model["value"] == 'Manufacturing':
            social_media_per_model[0] += 1
        elif records[i].business_model["value"] == 'Services':
            social_media_per_model[1] += 1
        elif records[i].business_model["value"] == 'Non Profit':
            social_media_per_model[2] += 1
        elif records[i].business_model["value"] == 'Wholesale':
            social_media_per_model[3] += 1
        elif records[i].business_model["value"] == 'Retail':
            social_media_per_model[4] += 1
        elif records[i].business_model["value"] == 'Education':
            social_media_per_model[5] += 1
        elif records[i].business_model["value"] == 'Government':
            social_media_per_model[6] += 1
social_media_per_model = normalize_to_100(social_media_per_model)
plt.figure(figsize=(6, 6))
plt.pie(social_media_per_model, labels=models, autopct='%1.1f%%', startangle=90, colors=['gold', 'yellowgreen', 'purple', 'lightcoral', 'skyblue', 'red', 'darkgreen'])
plt.axis('equal')
plt.title('Share of social media presence vs. bussines model')
plt.show()


"""
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

""" df['keywords'] = df['keywords'].astype(str)
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
"""