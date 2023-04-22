import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def get_valid_input(prompt, min_value, max_value):
    while True:
        try:
            value = int(input(prompt))
            if min_value <= value <= max_value:
                return value
            else:
                print(f"Please enter a value between {min_value} and {max_value}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Generate synthetic data: age, BMI, blood pressure, diabetes (0 = No, 1 = Yes)
np.random.seed(42)
ages = np.random.randint(20, 70, size=100)
bmis = np.random.randint(18, 41, size=100)
bps = np.random.randint(110, 170, size=100)
diabetes = np.random.randint(0, 2, size=100)

data = np.column_stack((ages, bmis, bps, diabetes))

X = data[:, :3]  # Features (age, BMI, blood pressure)
y = data[:, 3]  # Target (diabetes)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gaussian Naive Bayes model
gnb = GaussianNB()

# Train the model
gnb.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gnb.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Get user input for a new person's data
user_age = get_valid_input("Enter age (20-70): ", 20, 70)
user_bmi = get_valid_input("Enter BMI (18-40): ", 18, 40)
user_bp = get_valid_input("Enter blood pressure (110-170): ", 110, 170)

# Predict the probability of diabetes for the new person
new_person = np.array([[user_age, user_bmi, user_bp]])
prob_diabetes = gnb.predict_proba(new_person)
print(f"Probability of diabetes for the new person: {prob_diabetes[0][1]:.2f}")
