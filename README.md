# Recipe Recommendation System

## Overview
The **Recipe Recommendation System** suggests recipes based on the ingredients provided by the user. It uses **TF-IDF Vectorization** and **Cosine Similarity** to find the best-matching recipe from a dataset.

## Features
- Takes user input of available ingredients.
- Uses **machine learning** to find the closest matching recipe.
- Displays the recipe name, required ingredients, and instructions.
- Works with a CSV dataset of recipes.

## Requirements
Ensure you have Python installed along with the following dependencies:
```
pandas
scikit-learn
```
Install missing dependencies using:
```
pip install pandas scikit-learn
```

## Dataset Format (recipes.csv)
The dataset should be a CSV file containing the following columns:
```
recipe_name,ingredients,instructions
"Spaghetti Bolognese","pasta, tomato, beef, garlic, onion, olive oil","Cook pasta, prepare sauce, mix and serve."
```

## Usage
1. Place `recipes.csv` in the project directory.
2. Run the script:
```
python recipe_recommendation.py
```
3. Enter ingredients when prompted (comma-separated).
4. The best-matching recipe will be displayed.

## Example Input
```
Enter ingredients you have (comma-separated): pasta, tomato, beef
```

## Example Output
```
Recommended Recipe: Spaghetti Bolognese
Ingredients: pasta, tomato, beef, garlic, onion, olive oil
Instructions: Cook pasta, prepare sauce, mix and serve.
```

## Notes
- Ensure the `recipes.csv` file exists and is properly formatted.
- The system uses **text processing techniques** to compare ingredients effectively.

## License
This project is open-source and free to use for educational and personal purposes.
