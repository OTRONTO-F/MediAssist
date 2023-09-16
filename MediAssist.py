# Import the regular expression library
import re
# Import the pandas library
import pandas as pd
# Import the text-to-speech library
import pyttsx3
# Import preprocessing functions from scikit-learn
from sklearn import preprocessing
# Import DecisionTreeClassifier and _tree
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np  # Import the numpy library
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import cross_val_score function
from sklearn.model_selection import cross_val_score
# Import Support Vector Classifier
from sklearn.svm import SVC
# Import the CSV module
import csv
# Import the smtplib library for sending emails
import smtplib
# Import MIMEText for creating email messages
from email.mime.text import MIMEText
# Import warnings to disable DeprecationWarnings
import warnings

# Disable DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load training and testing data
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y

# Group reduced data
reduced_data = training.groupby(training['prognosis']).max()

# Encode string labels to numerical values
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)

# Prepare the testing data
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

# Create a Decision Tree classifier
clf1 = DecisionTreeClassifier()

# Fit the Decision Tree model with the training data
clf = clf1.fit(x_train, y_train)

# Evaluate the Decision Tree model using cross-validation
scores = cross_val_score(clf, x_test, y_test, cv=3)
print("Cross-validation scores:", scores.mean())

# Create an SVM classifier
model = SVC()

# Fit the SVM model with the training data
model.fit(x_train, y_train)

# Print the accuracy of the SVM model on the testing data
print("SVM accuracy:", model.score(x_test, y_test))

# Import feature importance values
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

# Initialize text-to-speech engine


def readn(nstr):
    # ใช้ Regular Expression (regex) เพื่อลบตัวอักษร '_' ออกจากข้อความ
    cleaned_text = re.sub('_', ' ', nstr)

    engine = pyttsx3.init()
    engine.setProperty(
        'voice', "english+f5")
    engine.setProperty('rate', 180)
    engine.say(cleaned_text)
    engine.runAndWait()
    engine.stop()


# Create dictionaries for symptom severity, description, and precautions
severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

symptoms_dict = {}

# Initialize symptom dictionary
for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index

# Calculate severity and assess symptom condition


def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum = sum + severityDictionary[item]
    print("\n\n\n\n\n\n\n\n----------------------------------------------------------------------------------------")
    print("Overall Severity of Disease: ")
    readn("Overall Severity of Disease: ")
    if sum <= 4:
        print("\nYou may have a mild case of the disease.")
        readn("You may have a mild case of the disease.")
    elif sum <= 7:
        print("\nYou may have a moderate case of the disease.")
        readn("You may have a moderate case of the disease.")
    else:
        print("\nYou may have a severe case of the disease.")
        readn("You may have a severe case of the disease.")
    print("\nSeverity of your symptoms: ", sum)
    readn("Severity of your symptoms: " + str(sum))
    print("\nYou may have the following diseases: ")
    readn("You may have the following diseases: ")
    if (sum * days) / (len(exp) + 1) > 13:
        print("\nIt is recommended to consult a doctor.")
        readn("It is recommended to consult a doctor.")
    else:
        print("\nIt might not be that serious, but you should take precautions.")
        readn("It might not be that serious, but you should take precautions.")

# Load disease descriptions from CSV


def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)

# Load symptom severity from CSV


def getSeverityDict():
    global severityDictionary
    with open('MasterData/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _diction = {row[0]: int(row[1])}
            severityDictionary.update(_diction)

# Load precaution information from CSV


def getprecautionDict():
    global precautionDictionary
    with open('MasterData/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)

# Get user information and introduce the chatbot


def getInfo():
    print("\n\n----------------------------------------------------------------------------------------")
    print("----------------------------------------MediAssist--------------------------------------")
    print("----------------------------------------------------------------------------------------")
    readn("Hello! Wellcome to MediAssist")
    print("\nPlease Enter Your Name: ", end="->\t")
    readn("Please Enter Your Name: ")
    name = input("")
    print("\nHello, ", name)
    readn("Hello, " + name)

# Check for symptom pattern


def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if (len(pred_list) > 0):
        return 1, pred_list
    else:
        return 0, []

# Predict disease using decision tree model


def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])

# Print predicted diseases


def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))

# Convert decision tree to code


def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        print("\nEnter the symptom you are experiencing (Press enter to view details.): ", end="->  ")
        readn("Enter the symptom you are experiencing (Press enter to view details.): ")
        disease_input = input("")
        conf, cnf_dis = check_pattern(chk_dis, disease_input)
        if conf == 1:
            print("\nSearches related to input: ")
            readn("Searches related to input: ")
            for num, it in enumerate(cnf_dis):
                print(num, ")", it)
            if num != 0:
                while True:
                    try:
                        print(
                            f"\nSelect the one you meant (0 - {num}):  ", end="")
                        readn(f"Select the one you meant (0 - {num}):  ")
                        conf_inp = int(input(""))
                        break
                    except ValueError:
                        print("Please enter a valid number.")
                        readn("Please enter a valid number.")
            else:
                conf_inp = 0

            disease_input = cnf_dis[conf_inp]
            break
        else:
            print("Enter a valid symptom.")
            readn("Enter a valid symptom.")

    while True:
        try:
            readn("How many days have you been experiencing this symptom? : ")
            num_days = int(
                input("\nHow many days have you been experiencing this symptom? : "))
            break
        except:
            print("Please enter a valid number.")
            readn("Please enter a valid number.")

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero(
            )]
            print("\nAre you experiencing any of the following symptoms?")
            readn("Are you experiencing any of the following symptoms?")
            symptoms_exp = []
            for syms in list(symptoms_given):
                inp = ""
                print("\n", syms, "?         (yes/no): ", end='')
                readn(syms + "?")
                while True:
                    inp = input("")
                    if (inp == "yes" or inp == "no"):
                        break
                    else:
                        print("\nPlease provide a valid answer (yes/no): ", end="")
                        readn("Please provide a valid answer (yes/no): ")
                if (inp == "yes"):
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            calc_condition(symptoms_exp, num_days)
            if (present_disease[0] == second_prediction[0]):
                print("\nYou may have ", present_disease[0])
                readn("You may have " + present_disease[0])
                print("\n", description_list[present_disease[0]])

            else:
                print("\nYou may have ",
                      present_disease[0], "or ", second_prediction[0])
                readn("You may have " +
                      present_disease[0] + "or " + second_prediction[0])
                print("\n", description_list[present_disease[0]])
                print("\n", description_list[second_prediction[0]])

            # Display precautions
            if present_disease[0] in precautionDictionary:
                print("\nPrecautions:")
                readn("Precautions:")
                precautions = precautionDictionary[present_disease[0]]
                for precaution in precautions:
                    print(f"- {precaution}")
                    readn(precaution)

            # Ask if the user wants to receive results via email
            send_email = ask_for_email()
            if send_email:
                readn("Please enter your email address: ")
                email = input("\nPlease enter your email address: ")
                send_email_results(
                    email,
                    [present_disease[0], description_list[present_disease[0]]],
                    symptoms_exp,
                    present_disease[0],
                    description_list[present_disease[0]]
                )

    recurse(0, 1)


# Function to ask if user wants to receive results via email


def ask_for_email():
    while True:
        readn("Do you want to receive the results via email? (yes/no): ")
        send_email = input(
            "\nDo you want to receive the results via email? (yes/no): ").strip().lower()

        if send_email == 'yes':
            return True
        elif send_email == 'no':
            return False
        else:
            print("Please enter 'yes' or 'no'.")
            readn("Please enter 'yes' or 'no'.")

# Function to send results via email


def send_email_results(email, results, user_symptoms, predicted_disease, disease_description):
    try:
        smtp_server = "smtp.elasticemail.com"
        smtp_port = 2525
        smtp_username = "thirayutsodaying@gmail.com"
        smtp_password = "B9C7CEE1AEE5F8DC02B30AA4E70DB341ABFB"

        subject = "MediAssist Results"
        message = f"Hello,\n\n"
        message += f"Thank you for using MediAssist to assess your symptoms.\n\n"
        message += f"Here are the results based on your input:\n\n"

        # Add user's symptoms
        message += f"Symptoms Reported by You:\n"
        for symptom in user_symptoms:
            message += f"- {symptom}\n"
        message += "\n"

        # Add predicted disease
        message += f"Predicted Disease:\n"
        message += f"- {predicted_disease}\n\n"

        # Add disease description
        message += f"Disease Description:\n"
        message += f"{disease_description}\n\n"

        # Add precautions
        if predicted_disease in precautionDictionary:
            precautions = precautionDictionary[predicted_disease]
            message += f"Precautions for {predicted_disease}:\n"
            for precaution in precautions:
                message += f"- {precaution}\n"
            message += "\n"

        message += f"We recommend consulting a healthcare professional for further evaluation and treatment.\n\n"
        message += f"Thank you for using MediAssist!\n\n"

        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = smtp_username
        msg['To'] = email

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(smtp_username, [email], msg.as_string())
        server.quit()
        print("\nResults sent to your email successfully!")
        readn("Results sent to your email successfully!")
    except Exception as e:
        print("\nError sending email:", str(e))
        readn("Error sending email:")


# Load severity, description, and precaution data
getSeverityDict()
getDescription()
getprecautionDict()
getInfo()

# Start the chatbot
tree_to_code(clf, cols)
print("\n------------------------------------------------------------------------------------------------------------------------------------------------------")
print("----------------------------------------------------Thank you for using MediAssist. Goodbye!----------------------------------------------------------")
print("------------------------------------------------------------------------------------------------------------------------------------------------------")
readn("Thank you for using MediAssist. Goodbye!")
