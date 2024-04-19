
import pandas

data = pandas.read_parquet("data/part-00000-ac5c3a2a-c89d-4817-8a00-1ba717b8f279-c000.snappy.parquet", engine = "pyarrow")
# print(data.dtypes)

data.to_csv('company_data.csv', index=False)