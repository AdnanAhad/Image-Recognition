import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample text data
text_data = "Your input text here. It can be multiple sentences."

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_data])
total_words = len(tokenizer.word_index) + 1

# Create input sequences and labels
input_sequences = []
for line in text_data.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[: i + 1]
        input_sequences.append(n_gram_sequence)

# Pad sequences for equal input length
max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(
    input_sequences, maxlen=max_sequence_length, padding="pre"
)

# Create predictors and labels
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Build the model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(
            total_words, 100, input_length=max_sequence_length - 1
        ),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(total_words, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=100, verbose=1)


# Function to generate text
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_len - 1, padding="pre"
        )
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


# Function to find the closest match from the trained data
def find_closest_match(input_text, text_data):
    distances = []
    for line in text_data.split("\n"):
        distances.append(tf.edit_distance(input_text, line).numpy())
    closest_match_index = np.argmin(distances)
    return text_data.split("\n")[closest_match_index]


# Take user input
user_input = input("Enter seed text: ")

# Find closest match from trained data
closest_match = find_closest_match(user_input, text_data)

# Generate text based on the closest match
generated_text = generate_text(closest_match, 10, model, max_sequence_length)
print(generated_text)
