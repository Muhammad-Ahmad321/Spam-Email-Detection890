Spam Email Detection App
This project demonstrates a simple Spam Email Detection web application using Streamlit and Naive Bayes Classifier. The app takes in an email message as input and predicts whether it is spam or not spam based on the content of the email.

Features:
Input email text for classification.
Predict if the email is spam or not.
Displays the accuracy of the model on test data.
Dataset:
The dataset used for training the model is a Spam vs Ham email dataset with labeled messages (spam or ham). The data includes:

Category: Whether the email is spam or ham.
Message: The content of the email.
The dataset is split into training and testing sets, and the model is trained using Multinomial Naive Bayes with a CountVectorizer to convert the text into numerical features.

Model:
Multinomial Naive Bayes: Chosen due to its effectiveness in text classification where the data is discrete.
CountVectorizer: Transforms the text data into a matrix of token counts.
Installation:
Clone the repository:

bash
Copy code
git clone <git@github.com:Muhammad-Ahmad321/Spam-Email-Detection890.git>
Install the required libraries:

bash
Copy code
pip install -r requirements.txt
Download the spam email dataset from Kaggle or Emails Dataset, and place it in the project directory.

Run the Streamlit app:

bash
Copy code
streamlit run app.py
How to Use:
Enter an email message into the text box.
Click the "Predict" button to classify the email as Spam or Not Spam.
Check the "Show model accuracy" option to see the accuracy of the trained model on the test data.
Dependencies:
Python 3.x
pandas
numpy
scikit-learn
streamlit
Acknowledgements:
Dataset from Kaggle: Email Spam Classification Dataset
License:
This project is licensed under the MIT License.