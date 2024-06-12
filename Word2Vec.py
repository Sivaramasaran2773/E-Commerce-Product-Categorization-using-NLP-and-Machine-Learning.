import pandas as pd
import numpy as np
import gensim.downloader as api
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
import re
from sklearn.decomposition import PCA

# Load normalized data
data = pd.read_csv(r"C:\Users\Dell\Downloads\normalized_data.csv")

# Download Word2Vec model
word2vec_model = api.load("word2vec-google-news-300")
label_dict = {0: 'Class 0', 1: 'Class 1', 2: 'Class 2', 3: 'Class 3'}  # Adjust according to your actual class labels and names

# Some useful functions for Word2Vec
def get_average_word2vec(tokens_list, model, generate_missing=False, k=300):
    if len(tokens_list) < 1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [model[word] if word in model else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [model[word] if word in model else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(model, tokens, generate_missing=False):
    embeddings = []
    for token in tokens:
        embeddings.append(get_average_word2vec(token.split(), model, generate_missing=generate_missing))
    return embeddings

def plot_embedding(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    unique_labels = np.unique(y)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        idx = y == label
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1], label=label_dict[label], color=color, alpha=0.5)
    
    ax.legend()
    plt.title('PCA of Word2Vec Embeddings')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

# Create tokens column from the description
data['tokens'] = data['description'].apply(lambda x: ' '.join(re.findall(r'\w+', x.lower())))

# Split data into train and test sets (70:30 ratio)
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Word2Vec embedding
X_train_embed = get_word2vec_embeddings(word2vec_model, train_data['tokens'])
X_test_embed = get_word2vec_embeddings(word2vec_model, test_data['tokens'])

y_train = train_data['label']
y_test = test_data['label']

# Plot the embeddings
plot_embedding(np.array(X_train_embed), y_train)

# Models to be trained
models = {
    "Linear SVM": SVC(kernel='linear', verbose=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "LightGBM": lgb.LGBMClassifier(objective='multiclass', num_class=len(np.unique(y_train))),
    "XGBoost": xgb.XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y_train)))
}

# Train and evaluate each model
results = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    
    if model_name in ["LightGBM", "XGBoost"]:
        model.fit(X_train_embed, y_train)
        y_pred_train = model.predict(X_train_embed)
        y_pred_test = model.predict(X_test_embed)
    else:
        model.fit(X_train_embed, y_train)
        y_pred_train = model.predict(X_train_embed)
        y_pred_test = model.predict(X_test_embed)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    precision = precision_score(y_test, y_pred_test, average='weighted')
    recall = recall_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')
    
    test_classification_report = classification_report(y_test, y_pred_test, target_names=label_dict.values())
    
    results[model_name] = {
        "Training Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Classification Report": test_classification_report
    }

# Print results
for model, result in results.items():
    print("\nResults for", model)
    print("Training Accuracy:", result["Training Accuracy"])
    print("Test Accuracy:", result["Test Accuracy"])
    print("Precision:", result["Precision"])
    print("Recall:", result["Recall"])
    print("F1 Score:", result["F1 Score"])
    print("Test Classification Report:")
    print(result["Classification Report"])
