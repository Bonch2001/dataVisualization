import requests
import pandas

class DataRecord:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

df = pandas.read_parquet("data/part-00000-ac5c3a2a-c89d-4817-8a00-1ba717b8f279-c000.snappy.parquet", engine = "pyarrow")
print(df.dtypes)

# print(data)

records = [DataRecord(**row.to_dict()) for index, row in df.iterrows()]

def generate_insight(prompt, model="davinci-002", max_tokens=100):
    api_key = 'your-gpt-key'  # Replace 'your-gpt-key' with your actual API key
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,        
        "prompt": prompt,       
        "max_tokens": max_tokens,
    }
    url = f'https://api.openai.com/v1/completions' 
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Example usage
prompt = ""
insight = generate_insight(prompt)
print(insight['choices'][0]['text'].strip())
