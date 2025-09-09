import pandas as pd

# Load CSV
df = pd.read_csv('sentences.csv')

# Define chunk size (number of sentences per chunk)
chunk_size = 50

# Function to split list into chunks
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield ' '.join(lst[i:i+n])

# Prepare a new DataFrame for concatenated chunks
concatenated_rows = []

# Group by 'source' and create chunks
for source, group in df.groupby('source'):
    texts = group['text'].tolist()
    for chunk in chunks(texts, chunk_size):
        concatenated_rows.append({'source': source, 'text': chunk})

# Create new DataFrame
df_chunks = pd.DataFrame(concatenated_rows)

# Save to new CSV
df_chunks.to_csv('concatenated_chunks.csv', index=False)

print(df_chunks)

