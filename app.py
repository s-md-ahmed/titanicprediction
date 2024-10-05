import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the preprocessed DataFrame (X is features, y is the target variable)
df = pd.read_csv('train.csv')

# Drop unnecessary columns
dropped_columns = ["PassengerId", "Name", "Ticket", "Cabin"]
df.drop(dropped_columns, inplace=True, axis=1)

# Map 'Sex' to 0 and 1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Map 'Embarked'
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Separate features and target variable
X = df.drop("Survived", axis=1)  # Features
y = df["Survived"]  # Target variable

# Fill missing values in the 'Age' column with the mean age
mean_age = X.loc[X['Age'].notnull(), 'Age'].mean()
X['Age'] = X['Age'].fillna(mean_age)

# Fill missing values for other relevant columns (if applicable)
X['Fare'] = X['Fare'].fillna(X['Fare'].mean())  # Example: Fill missing Fare with mean
X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])  # Fill missing Embarked with mode

# Check for any remaining NaN values
if X.isnull().sum().any():
    st.error("There are still missing values in the input data.")
else:
    # Apply RobustScaler to features
    robust_scaler = RobustScaler()
    X_scaled = robust_scaler.fit_transform(X)

    # Apply MinMaxScaler to features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_scaled)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize the Support Vector Machine Classifier with fixed hyperparameters
    svm_classifier = SVC(kernel='rbf', C=1, gamma=5, random_state=42)

    # Fit the model on the training set
    svm_classifier.fit(X_train, y_train)

    # Streamlit App
    st.title("Welcome to my Titanic Survival Prediction App")

    # Custom CSS to change slider and button colors
    css = """
        <style>
            .css-1j4b5fy-Badge {
                background-color: aqua;
            }
            .css-1eh7lfl-Badge {
                background-color: green;
            }
        </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # Form for user input
    with st.form("user_input_form"):
        # Create input fields for features
        sex = st.selectbox("Sex", ["Male", "Female"])
        age = st.slider("Age", int(X['Age'].min()), int(X['Age'].max()), int(mean_age))
        pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
        sibsp = st.slider("Number of Siblings/Spouses (SibSp)", int(X['SibSp'].min()), int(X['SibSp'].max()), int(X['SibSp'].mean()))
        parch = st.slider("Number of Parents/Children (Parch)", int(X['Parch'].min()), int(X['Parch'].max()), int(X['Parch'].mean()))
        fare = st.slider("Fare", float(X['Fare'].min()), float(X['Fare'].max()), float(X['Fare'].mean()))
        embarked = st.selectbox("Port of Embarkation (Embarked)", ['S', 'C', 'Q'])
        # Submit button
        submit_button = st.form_submit_button("Submit Prediction")

    # Map user input to match the model features
    sex_mapping = {'Male': 0, 'Female': 1}
    sex = sex_mapping[sex]
    embark_mapping = {'S': 0, 'C': 1, 'Q': 2}
    embarked = embark_mapping[embarked]

    # Create a DataFrame with user input
    user_input = pd.DataFrame({
        'Sex': [sex],
        'Age': [age],
        'Pclass': [pclass],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })

    # Ensure that the order of columns in user_input matches the order of columns in X_train
    user_input = user_input[X.columns]

    # Apply preprocessing to user input
    user_input_scaled = robust_scaler.transform(user_input)
    user_input_scaled = scaler.transform(user_input_scaled)

    # Make predictions and display only when the form is submitted
    if submit_button:
        # Make predictions on user input
        prediction = svm_classifier.predict(user_input_scaled)

        # Display prediction
        st.subheader("Prediction")
        st.write("Passenger survived" if prediction[0] == 1 else "Passenger did not survive")

        # Display classification report for the training set
        # Display classification report for the training set
        st.subheader("Classification Report (Training Set)")
        y_train_pred = svm_classifier.predict(X_train)
        classification_rep_train = classification_report(y_train, y_train_pred, output_dict=True)

# Convert to DataFrame and round to 2 decimal places
        classification_rep_train_df = pd.DataFrame(classification_rep_train).transpose()
        classification_rep_train_df = classification_rep_train_df.round(2)

# Display the table in Streamlit
        st.table(classification_rep_train_df)

# Display classification report for the test set
        st.subheader("Classification Report (Test Set)")
        y_test_pred = svm_classifier.predict(X_test)
        classification_rep_test = classification_report(y_test, y_test_pred, output_dict=True)

# Convert to DataFrame and round to 2 decimal places
        classification_rep_test_df = pd.DataFrame(classification_rep_test).transpose()
        classification_rep_test_df = classification_rep_test_df.round(2)

# Display the table in Streamlit
        st.table(classification_rep_test_df)
