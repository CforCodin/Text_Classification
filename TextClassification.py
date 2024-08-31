from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
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
st.title("Text Classification")
st.subheader("Enter 4 text Documents to classify")
new_documents = []
for i in range(4):
    doc =  st.text_input(f"Enter document {i+1}", key=f"text_input_{i}")
    new_documents.append(doc)
    st.write(f"Document {i+1}: ", new_documents[i])

new_documents = [doc for doc in new_documents if isinstance(doc, str) and doc.strip()]
if new_documents:  # Only transform if there are valid documents
    new_documents_tfidf = vectorizer.transform(new_documents)

    # Step 2: Use the trained classifier to predict the class of these new documents
    predictions = clf.predict(new_documents_tfidf)
    predictions2 = clf2.predict(new_documents_tfidf)

    # Display the predictions
    if st.button("Classify using Decision Tree"):
        st.markdown("### Predictions from Decision Tree")
        for doc, category in zip(new_documents, predictions):
            st.success(f"**Document:** {doc}")
            st.info(f"**Predicted Category:** {newsgroups_train.target_names[category]}")
            st.markdown("---")  # Separator

    if st.button("Classify using Random Forest"):
        st.markdown("### Predictions from Random Forest")
        for doc, category in zip(new_documents, predictions2):
            st.success(f"**Document:** {doc}")
            st.info(f"**Predicted Category:** {newsgroups_train.target_names[category]}")
            st.markdown("---")  # Separator
else:
    st.write("Please enter valid text documents for classification.")