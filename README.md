# SMS Spam Classifier

This Python script trains a Naive Bayes classifier to distinguish between spam and legitimate (ham) SMS messages. It then allows you to classify new messages interactively.

## Setup

1.  **Clone the repository (if applicable) or download `spam_classifier.py`.**
2.  **Install dependencies:**
    Open your terminal or command prompt and run:
    ```bash
    pip install -r requirements.txt
    ```
    This will install all necessary libraries. The script also uses NLTK resources (`stopwords` and `wordnet`), which it will attempt to download on first run if they are not found.

## Dataset Format

You need a CSV file containing SMS messages. The script expects the following:
*   Two columns.
*   The first column should contain the labels: 'spam' or 'ham'.
*   The second column should contain the text of the SMS message.
*   **Crucially, during the data loading phase in `spam_classifier.py`, the script assumes these columns are named 'label' and 'text' respectively.** If your CSV has different names, you will need to modify the `pd.read_csv()` line and relevant column accesses in `spam_classifier.py` or ensure your CSV header matches these names. For example:
    ```csv
    label,text
    ham,"Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."
    spam,"Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
    # ... and so on
    ```

## Running the Script

1.  **Navigate to the script's directory in your terminal.**
2.  **Run the script:**
    ```bash
    python spam_classifier.py --dataset /path/to/your/spam_data.csv
    ```
    *   Replace `/path/to/your/spam_data.csv` with the actual path to your dataset.
    *   If your dataset is named `spam_dataset.csv` and is in the same directory as the script, you can simply run:
        ```bash
        python spam_classifier.py
        ```

3.  **Output:**
    *   The script will first print some information about the loaded dataset (shape, head, label distribution).
    *   Then, it will show the first 5 processed messages.
    *   Next, it will print "Model trained successfully!"
    *   Evaluation metrics (Accuracy, Precision, Recall, F1-score) for the test set will be displayed.
    *   A confusion matrix plot will be saved as `confusion_matrix.png` in the same directory.
    *   Finally, you will be prompted: `Enter a message to classify (or type 'exit' to quit):`

## Classifying New Messages

*   Type or paste the message you want to classify and press Enter.
*   The script will output its prediction (e.g., `Prediction: Spam` or `Prediction: Ham`).
*   To stop the script, type `exit` and press Enter.
