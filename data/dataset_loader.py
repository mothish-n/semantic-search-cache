import os

def load_newsgroups(dataset_path):

    documents = []
    labels = []

    for category in os.listdir(dataset_path):

        category_path = os.path.join(dataset_path, category)

        if not os.path.isdir(category_path):
            continue

        for file in os.listdir(category_path):

            file_path = os.path.join(category_path, file)

            try:
                with open(file_path, "r", encoding="latin1") as f:
                    text = f.read()

                documents.append(text)
                labels.append(category)

            except:
                continue

    return documents, labels