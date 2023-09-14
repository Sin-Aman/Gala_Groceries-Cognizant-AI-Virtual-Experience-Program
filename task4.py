import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load data from CSV
def load_data(csv_path):
    """
    Load data from a CSV file into a Pandas DataFrame.

    Parameters:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the data.
    """
    df = pd.read_csv(csv_path)
    return df


# Split data into target variable and predictor variables
def create_target_and_predictors(data: pd.DataFrame, target: str = "estimated_stock_pct"):
    """
    Split data into predictor variables (X) and the target variable (y).

    :param data: pd.DataFrame, dataframe containing the data.
    :param target: str (optional), the target variable to predict.

    :return X: pd.DataFrame, predictor variables.
    :return y: pd.Series, target variable.
    """
    if target not in data.columns:
        raise ValueError(f"Target variable '{target}' is not present in the data.")
    
    X = data.drop(columns=[target])
    y = data[target]
    return X, y

# Train the algorithm using cross-validation
def train_algorithm_with_cross_validation(X: pd.DataFrame, y: pd.Series, K=10, SPLIT=0.75):
    """
    Train a Random Forest Regressor model with K-fold cross-validation.

    :param X: pd.DataFrame, predictor variables.
    :param y: pd.Series, target variable.
    :param K: int, number of folds for cross-validation.
    :param SPLIT: float, fraction of data used for training (0.0 to 1.0).

    :return: None
    """
    accuracy = []

    for fold in range(K):
        # Instantiate the model and scaler
        model = RandomForestRegressor()
        scaler = StandardScaler()

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=SPLIT, random_state=42)

        # Scale X data
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the model
        trained_model = model.fit(X_train, y_train)

        # Generate predictions on the test set
        y_pred = trained_model.predict(X_test)

        # Compute Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        accuracy.append(mae)
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")

    # Compute and display the average MAE
    avg_mae = sum(accuracy) / len(accuracy)
    print(f"Average MAE: {avg_mae:.2f}")

if __name__ == "__main__":
    # Replace 'path/to/your/csv_file.csv' with the actual path to your CSV file
    csv_file_path = "C:\\Users\\amans\\Downloads\\Gola groceries(Cognizant)\\Task3\\sales.csv"

    # Load the data
    data = load_data(csv_file_path)

    # Split data into predictor variables (X) and target variable (y)
    X, y = create_target_and_predictors(data)

    # Train the model with cross-validation and print the average MAE
    train_algorithm_with_cross_validation(X, y)