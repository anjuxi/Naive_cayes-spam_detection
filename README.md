# Naive Bayes Spam Detection

A classic Naive Bayes classifier that labels email as spam or ham based on word frequencies. The project includes training, evaluation, and a few prediction examples.

## Dataset

- **File**: `spam_ham_dataset.csv`, from https://www.kaggle.com/datasets/venky73/spam-mails-dataset/data
- **Size**: 5,171 emails
  - Ham: 3,672
  - Spam: 1,499
- **Fields**:
  - `label`: email label (ham/spam)
  - `text`: email body
  - `label_num`: numeric label (0=ham, 1=spam)

## Implementation

### Core Algorithm

**Naive Bayes classifier**

- Computes posterior probabilities with Bayes' theorem
- Assumes features (words) are independent
- Uses Laplace smoothing for unseen words
- Uses log probabilities to avoid numerical underflow

### Text Preprocessing

1. Lowercase
2. Remove non-letter characters
3. Tokenize
4. Drop short tokens (length ≤ 2)

### Key Techniques

| Technique | Notes |
| ------------ | ---------------------------------------- |
| Laplace smoothing | Handles words not seen in training, avoids zero probabilities |
| Log probabilities | Prevents underflow from multiplying many small probabilities |
| Priors | Based on the spam/ham ratio in the training set |

## Project Structure

```
spam_detection/
├── spam_ham_dataset.csv      # dataset
├── naive_bayes_spam.ipynb    # main notebook
├── LICENSE                   # license
└── README.md                 # this file
```

## Usage

```bash
jupyter notebook naive_bayes_spam.ipynb
```

## Performance

With an 80/20 train/test split:

| Metric | Value |
| ------------------ | ---------- |
| Accuracy | **98.07%** |
| Precision | **96.27%** |
| Recall | **96.93%** |
| F1 score | **96.60%** |

## Example Predictions

The project includes four sample emails:

| Email content | Prediction |
| ---------------------------------------------------- | ---------- |
| "Congratulations! You have won a $1000 gift card..." | Spam ✓ |
| "Meeting schedule for next week..." | Ham ✓ |
| "Get rich quick! Buy cheap viagra now!!!" | Spam ✓ |
| "Project update - Q3 financial report..." | Ham ✓ |

## How It Works

Naive Bayes is based on Bayes' theorem:

```
P(class|doc) = P(doc|class) × P(class) / P(doc)
```

Where:

- **Prior** P(class): the proportion of each class in the training set
- **Likelihood** P(doc|class): the probability of the document given the class
- **Posterior** P(class|doc): the probability that the document belongs to the class

Assuming word independence, the document probability factors into a product of word probabilities:

```
P(doc|class) = ∏ P(word|class)
```

## License

**MIT**

---

## Author

**Dr. Ailan Anjuxi (谙弆悕)**

- **Email**: anjuxi.ME@outlook.com
- **SIP**: sip:anjuxi@sip.linphone.org

---

*A classic application of Naive Bayes to text classification. Suitable as an introductory machine learning example.*
