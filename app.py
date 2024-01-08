import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load the preprocessed DataFrame (X is features, y is the target variable)
df = pd.read_csv(r'D:\titanic\train.csv')

# Drop unnecessary columns
dropped_columns = ["PassengerId", "Name", "Ticket", "Cabin", "Embarked"]
df.drop(dropped_columns, inplace=True, axis=1)

# Map 'Sex' to 0 and 1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Separate features and target variable
X = df.drop("Survived", axis=1)  # Features
y = df["Survived"]  # Target variable

# Fill missing values in the 'Age' column with the mean age
mean_age = X.loc[X['Age'].notnull(), 'Age'].mean()
X['Age'] = X['Age'].fillna(mean_age)

# Apply RobustScaler to features
robust_scaler = RobustScaler()
X_scaled = robust_scaler.fit_transform(X)

# Apply MinMaxScaler to features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_scaled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=5, random_state=42)

# Fit the model on the training set
rf_classifier.fit(X_train, y_train)

# Streamlit App
st.title("Welcome to my Titanic Survival Prediction App")

# Form for user input
with st.form("user_input_form"):
    # Create input fields for features
    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.slider("Age", float(X['Age'].min()), float(X['Age'].max()), float(mean_age))
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
    sibsp = st.slider("Number of Siblings/Spouses (SibSp)", int(X['SibSp'].min()), int(X['SibSp'].max()), int(X['SibSp'].mean()))
    parch = st.slider("Number of Parents/Children (Parch)", int(X['Parch'].min()), int(X['Parch'].max()), int(X['Parch'].mean()))
    fare = st.slider("Fare", float(X['Fare'].min()), float(X['Fare'].max()), float(X['Fare'].mean()))

    # Submit button
    submit_button = st.form_submit_button("Submit Prediction")

# Map user input to match the model features
sex_mapping = {'Male': 0, 'Female': 1}
sex = sex_mapping[sex]

# Create a DataFrame with user input
user_input = pd.DataFrame({
    'Sex': [sex],
    'Age': [age],
    'Pclass': [pclass],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare]
})

# Ensure that the order of columns in user_input matches the order of columns in X_train
user_input = user_input[X.columns]

# Apply preprocessing to user input
user_input_scaled = robust_scaler.transform(user_input)
user_input_scaled = scaler.transform(user_input_scaled)

# Make predictions and display only when the form is submitted
if submit_button:
    # Make predictions on user input
    prediction = rf_classifier.predict(user_input_scaled)

    # Display prediction
    st.subheader("Prediction")
    st.write("Survived" if prediction[0] == 1 else "Not Survived")

    # Display classification report for the training set
    st.subheader("Classification Report (Training Set)")
    y_train_pred = rf_classifier.predict(X_train)
    classification_rep_train = classification_report(y_train, y_train_pred, output_dict=True)
    st.table(pd.DataFrame(classification_rep_train).transpose())

    # Display classification report for the test set
    st.subheader("Classification Report (Test Set)")
    y_test_pred = rf_classifier.predict(X_test)
    classification_rep_test = classification_report(y_test, y_test_pred, output_dict=True)
    st.table(pd.DataFrame(classification_rep_test).transpose())
