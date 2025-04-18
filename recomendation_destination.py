import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
import torch
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Загрузка данных
df = pd.read_csv('europe_only.csv')
df["CATEGORIES"] = df["CATEGORIES"].apply(eval)

# Заполнение пропущенных значений
df['NAME'] = df['NAME'].fillna('')
df['DESTINATION'] = df['DESTINATION'].fillna('')
df['DESCRIPTION'] = df['DESCRIPTION'].fillna('')

# Предобработка текста
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  
    return text

df['PROCESSED_TEXT'] = (df['NAME'] + " " + df['DESTINATION'] + " " + df['DESCRIPTION']).apply(preprocess_text)

# Получение эмбеддингов
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

if os.path.exists("text_embeddings.npy"):
    text_embeddings = np.load("text_embeddings.npy")
else:
    text_embeddings = model.encode(df['PROCESSED_TEXT'].tolist(), show_progress_bar=True)
    np.save("text_embeddings.npy", text_embeddings)

# Преобразование категорий
mlb = MultiLabelBinarizer()
category_features = mlb.fit_transform(df['CATEGORIES'])

# TF-IDF для категорий
category_texts = [' '.join(cats) for cats in df['CATEGORIES']]
category_vectorizer = TfidfVectorizer(max_features=100)
category_tfidf = category_vectorizer.fit_transform(category_texts).toarray()

# Объединение признаков
combined_features = np.hstack([
    text_embeddings,
    category_features,
    category_tfidf
])

if os.path.exists("nearest_neighbors_model.pkl"):
    nn = joblib.load("nearest_neighbors_model.pkl")
else:
    nn = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
    nn.fit(combined_features)
    joblib.dump(nn, "nearest_neighbors_model.pkl")

def get_recommendations(query, top_k=5, category_filter=None):
    processed_query = preprocess_text(query)
    query_embedding = model.encode([processed_query])[0]
    
    if category_filter:
        valid_categories = [cat for cat in category_filter if cat in mlb.classes_]
        if not valid_categories:
            return pd.DataFrame()
        
        query_categories = mlb.transform([valid_categories])[0]
        query_category_tfidf = category_vectorizer.transform([' '.join(valid_categories)]).toarray()[0]
    else:
        query_categories = np.zeros(category_features.shape[1])
        query_category_tfidf = np.zeros(category_tfidf.shape[1])
    
    query_combined = np.hstack([
        query_embedding,
        query_categories,
        query_category_tfidf
    ]).reshape(1, -1)
    
    distances, indices = nn.kneighbors(query_combined, n_neighbors=top_k)
    
    recommendations = df.iloc[indices[0]].copy()
    recommendations['SIMILARITY'] = 1 - distances[0]
    
    return recommendations.sort_values('SIMILARITY', ascending=False)

# Сохранение моделей
torch.save(model.state_dict(), "text_model.pth")
np.save("category_mlb_classes.npy", mlb.classes_)
np.save("category_vectorizer_vocab.npy", category_vectorizer.vocabulary_)

# Примеры запросов
queries = [
    ("museums and historical places", ["museum", "history"]),
    ("night clubs and bars", ["Bars & Clubs", "Nightlife"]),
    ("nature and parks", ["Parks"]),
    ("family activities", None)
]

for query, categories in queries:
    recs = get_recommendations(query, category_filter=categories)
    if not recs.empty:
        print(f"\nRecommendations for: '{query}'")
        if categories:
            print(f"Categories: {', '.join(categories)}")
        print("-" * 60)
        
        unique_recs = recs.drop_duplicates(subset=['NAME']).head(5)
        
        for i, (_, row) in enumerate(unique_recs.iterrows(), 1):
            print(f"{i}. {row['NAME']} (Rating: {row['RATING']:.1f}, Similarity: {row['SIMILARITY']:.2f})")
        
        print("=" * 60 + "\n")
    else:
        print(f"\nNo results found for: '{query}'\n" + "=" * 60 + "\n")