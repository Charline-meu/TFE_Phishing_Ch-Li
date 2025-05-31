import re
from langdetect import detect, LangDetectException
import numpy as np

class EmailPreprocessor:
    def preprocess(self, text):
        if text == "-1":
            return np.nan  # Return NaN for invalid "-1" entries

        try:
            if detect(text) != 'en':
                return np.nan  # Return NaN for non-English entries
        except LangDetectException:
            return np.nan  # Return NaN when detection fails

        text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
        text = re.sub(r"\d+", "", text)  # Remove numbers
        text = re.sub(r"\S+@\S+\.\S+", "", text)  # Remove email addresses
        #text = re.sub(r"[^A-Za-z0-9\s]", "", text)  # Remove special characters
        text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
        return text

# Instantiate the class
preprocessor = EmailPreprocessor()

# Test examples
test_cases = [
    {
        "input": "Hello, please visit our website at https://example.com and contact us at info@example.com. Thank you!",
        "description": "English with URL and email"
    },
    {
        "input": "Bonjour, veuillez visiter notre site web à l'adresse https://exemple.fr. Merci!",
        "description": "French (non-English)"
    },
    {
        "input": "こんにちは、私たちのウェブサイトをご覧ください。ありがとう！",
        "description": "Japanese (non-English)"
    },
    {
        "input": "Order #12345 confirmed! Email: customer123@example.com. Visit http://ordertracker.com for details.",
        "description": "English with numbers, symbols, URL, and email"
    },
    {
        "input": "asdfghjklqwertyuiopzxcvbnm1234567890",
        "description": "Gibberish / detection fail likely"
    },
    {
        "input": "Just a quick reminder to submit the report by Friday.",
        "description": "Clean English"
    },
    {
        "input": "⚠️⚠️ REAL-TIME TRADER ACCESS REQUIRED ⚠️⚠️",
        "description": "Clean English with emojis"
    },
    {
        "input": "Si je commence ma phrase en français, but that I end my sentence in English. What will happen?",
        "description": "Clean English with emojis"
    },
]

# Run and display results
for i, test in enumerate(test_cases, 1):
    result = preprocessor.preprocess(test["input"])
    print(f"Test Case {i}: {test['description']}")
    print(f"Input: {test['input']}")
    print(f"Output: '{result}'")
    print("-" * 60)