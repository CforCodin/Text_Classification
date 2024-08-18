from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
# Load the dataset
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
ch = 0
print("Articles we are taining model on:")
for i in categories:
    if ch < 5:
        print(i)
    else:
        break
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))


X_train, X_test, Y_train, Y_test = train_test_split(newsgroups_train.data, newsgroups_train.target, test_size = 0.3, random_state= 42)
class_distribution = np.bincount(Y_train)
plt.bar(range(len(class_distribution)), class_distribution)
plt.xticks(range(len(class_distribution)), newsgroups_train.target_names, rotation = 45)
plt.title("Distribution of Classes in Training set")
plt.xlabel('Class')
plt.ylabel('Number of documents')
plt.show()

class_distribution = np.bincount(Y_test)
plt.bar(range(len(class_distribution)), class_distribution)
plt.xticks(range(len(class_distribution)), newsgroups_train.target_names, rotation = 45)
plt.title("Distribution of Classes in Test set")
plt.xlabel('Class')
plt.ylabel('Number of documents')
plt.show()


vectorizer = TfidfVectorizer(stop_words='english')
X_train_d = vectorizer.fit_transform(X_train)
X_test_d = vectorizer.transform(X_test)


clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_d, Y_train)
Y_prd = clf.predict(X_test_d)


from sklearn.ensemble import RandomForestClassifier

clf2 = RandomForestClassifier(random_state=42)
clf2.fit(X_train_d, Y_train)
Y_pdd = clf2.predict(X_test_d)


print("After Training from Decision Tree")
print("Accuracy:", accuracy_score(Y_test, Y_prd))
print("\n Classification Report: \n", classification_report(Y_test, Y_prd, target_names = newsgroups_test.target_names))

print("After Training from Random Forest")
print("Accuracy:", accuracy_score(Y_test, Y_pdd))
print("\n Classification Report: \n", classification_report(Y_test, Y_pdd, target_names = newsgroups_test.target_names))

# Example of new, unseen data
new_documents = [
    "The graphics on this computer are amazing.",
    "Atheism is the absence of belief in gods.",
    "Medicine has advanced significantly over the years.",
    "Atheism believe in the teachings of Humanity."
]

# Step 1: Transform the new documents using the existing vectorizer
new_documents_tfidf = vectorizer.transform(new_documents)

# Step 2: Use the trained classifier to predict the class of these new documents
predictions = clf.predict(new_documents_tfidf)
predictions2 = clf2.predict(new_documents_tfidf)

# Step 3: Display the predictions
print("Predictions from Decision Tree")
for doc, category in zip(new_documents, predictions):
    print(f"Document: {doc}")
    print(f"Predicted category: {newsgroups_train.target_names[category]}\n")

print("Predictions from Random Forest")
for doc, category in zip(new_documents, predictions2):
    print(f"Document: {doc}")
    print(f"Predicted category: {newsgroups_train.target_names[category]}\n")

