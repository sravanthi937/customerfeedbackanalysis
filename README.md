# Customer Feedback Analysis

This project analyzes customer feedback from Flipkart reviews to gain insights using Natural Language Processing (NLP) techniques. The goal is to classify sentiments and extract useful patterns that help understand customer satisfaction.

## 📝 Project Description

E-commerce platforms like Flipkart receive a vast amount of customer reviews. Manually going through these reviews is time-consuming and inefficient. This project uses Python and machine learning models to automate the sentiment analysis of these reviews and provide an output summary.

### Key Features:
- Loads and preprocesses customer review data
- Cleans text data (removal of stopwords, punctuation, etc.)
- Converts text into numerical features using TF-IDF
- Trains a machine learning model to classify sentiment (positive/negative)
- Exports predictions to a CSV file for easy viewing

---

## 📁 File Structure

```plaintext
customerfeedbackanalysis/
├── flipkart.csv        # Raw customer reviews dataset
├── model1.py           # Python script for preprocessing, training, and predicting
├── output.csv          # Output predictions of sentiments for each review
└── README.md           # Project documentation
⚙️ Requirements
Python 3.x

pandas

numpy

scikit-learn

nltk

Install dependencies using:

bash
Copy
Edit
pip install pandas numpy scikit-learn nltk
🚀 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/sravanthi937/customerfeedbackanalysis.git
cd customerfeedbackanalysis
Run the model script:

bash
Copy
Edit
python model1.py
Check the output.csv file for the prediction results.

📊 Sample Output
The output.csv file contains the original reviews along with the predicted sentiment label (e.g., Positive, Negative).

🔍 Future Improvements
Add support for more sentiment categories (e.g., Neutral)

Deploy the model as a web service using Flask or FastAPI

Integrate a dashboard for visualization of insights

Use deep learning techniques (e.g., LSTM, BERT)

📌 Dataset Source
The flipkart.csv file is a collection of scraped customer reviews. Please ensure data usage aligns with Flipkart's terms of service.

👩‍💻 Author
Sravanthi Akula
GitHub: sravanthi937
