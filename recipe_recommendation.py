import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_recipes(file_path):
    """Loads recipe data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Combines ingredients into a single text field for vectorization."""
    df['ingredients_combined'] = df['ingredients'].apply(lambda x: ' '.join(x.lower().split(',')))
    return df

def recommend_recipe(user_ingredients, df, vectorizer):
    """Recommends the best matching recipe based on cosine similarity."""
    user_ingredients = ' '.join(user_ingredients.lower().split(','))
    tfidf_matrix = vectorizer.fit_transform(df['ingredients_combined'])
    user_tfidf = vectorizer.transform([user_ingredients])
    
    similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    best_match_index = similarities.argmax()
    return df.iloc[best_match_index]

def main():
    file_path = "recipes.csv"  # Ensure this file exists
    df = load_recipes(file_path)
    df = preprocess_data(df)
    
    vectorizer = TfidfVectorizer()
    vectorizer.fit(df['ingredients_combined'])
    
    user_ingredients = input("Enter ingredients you have (comma-separated): ")
    recommended_recipe = recommend_recipe(user_ingredients, df, vectorizer)
    
    print(f"\nRecommended Recipe: {recommended_recipe['recipe_name']}")
    print(f"Ingredients: {recommended_recipe['ingredients']}")
    print(f"Instructions: {recommended_recipe['instructions']}")

if __name__ == "__main__":
    main()
