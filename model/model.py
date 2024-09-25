import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('./data/sales.csv')


# Step 1: Generate synthetic product data
def generate_product_data(n_samples=1000):
    categories = ['Electronics', 'Clothing', 'Home Decor', 'Sports', 'Beauty']
    product_names = [f'Product_{i}' for i in range(n_samples)]
    
    # Random product data
    category = [random.choice(categories) for _ in range(n_samples)]
    price = np.random.uniform(10, 500, n_samples)
    discount = np.random.uniform(0, 0.5, n_samples)  # Discounts between 0 and 50%
    sales_quantity = np.random.randint(1, 1000, n_samples)
    
    # Create data as a DataFrame
    data = pd.DataFrame({
        'product_name': product_names,
        'category': category,
        'price': price,
        'discount': discount,
        'sales_quantity': sales_quantity
    })
    
    return data

# Step 2: Preprocess data
def preprocess_data(df):
    df = pd.get_dummies(df, columns=['category'], drop_first=True)
    X = df.drop(columns=['product_name', 'price'])  # Feature matrix (exclude product_name and price)
    y = df['price']  # Target (price)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Step 3: Define models and AutoML with GridSearchCV
def apply_automl(X_train, X_test, y_train, y_test):
    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet()
    }
    
    # Parameter grid for hyperparameter tuning
    param_grid = {
        'Ridge': {'alpha': np.logspace(-4, 4, 50)},
        'Lasso': {'alpha': np.logspace(-4, 4, 50)},
        'ElasticNet': {'alpha': np.logspace(-4, 4, 50), 'l1_ratio': np.linspace(0, 1, 50)}
    }
    
    best_models = {}
    best_r2_scores = {}

    # Iterate through the models and perform GridSearchCV
    for model_name, model in models.items():
        print(f"Training {model_name} model...")
        grid_search = GridSearchCV(model, param_grid[model_name], cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        best_models[model_name] = best_model
        best_r2_scores[model_name] = r2
        print(f"{model_name} best R-squared: {r2}")
    
    return best_models, best_r2_scores

# Step 4: Execute the process
if __name__ == "__main__":
    # Generate synthetic product data
    product_data = generate_product_data()
    print("Generated Product Data (head):")
    print(product_data.head())

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(product_data)
    
    # Apply AutoML to experiment with Ridge, Lasso, and ElasticNet
    best_models, best_r2_scores = apply_automl(X_train, X_test, y_train, y_test)

    print("\nBest R-squared scores from models:")
    for model_name, r2_score_value in best_r2_scores.items():
        print(f"{model_name}: {r2_score_value}")
