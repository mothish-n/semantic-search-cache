from data.dataset_loader import load_newsgroups
from data.text_cleaner import clean_text

dataset_path = r"C:\Users\mothi\OneDrive\Desktop\semantic-search-cache\data\20_newsgroups"

docs, labels = load_newsgroups(dataset_path)

clean_docs = [clean_text(doc) for doc in docs]

print("Total documents:", len(clean_docs))
print("First category:", labels[0])
print("Cleaned preview:\n")
print(clean_docs[0][:500])