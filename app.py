import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# First of all, create a .env with your bot token
load_dotenv()

# Load the dataset
# Available in: https://www.kaggle.com/datasets/sahityasetu/paris-2024-olympics-faq Thank you Sahitya Setu!
df = pd.read_csv('paris-2024-faq.csv', sep=';', quotechar='"', escapechar='\\', encoding='utf-8', on_bad_lines='skip')

# Filter dataset by language, English and French are available in the dataset
df_en = df[df['lang'] == 'en']
df_fr = df[df['lang'] == 'fr']

# Vectorize questions for each language
vectorizer_en = TfidfVectorizer()
X_en = vectorizer_en.fit_transform(df_en['label'].tolist())

vectorizer_fr = TfidfVectorizer()
X_fr = vectorizer_fr.fit_transform(df_fr['label'].tolist())

# Function to find the best match based on the detected language
def find_best_match(user_question, lang):
    if lang == 'fr':
        user_input_vec = vectorizer_fr.transform([user_question])
        similarities = cosine_similarity(user_input_vec, X_fr).flatten()
        best_match_idx = similarities.argmax()
        return df_fr['body'].iloc[best_match_idx], df_fr['url'].iloc[best_match_idx]
    else:  # Default to English
        user_input_vec = vectorizer_en.transform([user_question])
        similarities = cosine_similarity(user_input_vec, X_en).flatten()
        best_match_idx = similarities.argmax()
        return df_en['body'].iloc[best_match_idx], df_en['url'].iloc[best_match_idx]

# Function to handle the /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Hello, I am an FAQ bot for the Paris 2024 Olympics. Ask me a question in English or French!')

# Function to handle user messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_question = update.message.text
    print(user_question)
    # Detect the language of the user's question
    try:
        lang = detect(user_question)
    except:
        lang = 'en'  # Default to English
    
    if lang == 'fr':
        await update.message.reply_text("Vous avez posé une question en français, je vais répondre en français.")
    else:
        await update.message.reply_text("You asked a question in English, I will respond in English.")

    # Find the best answer based on the detected language
    answer, url = find_best_match(user_question, lang)
    
    response = f"Answer: {answer}"
    if url:
        response += f"\nMore info: {url}" # URL for more information based on dataset
    
    await update.message.reply_text(response)

# Main function to set up the bot
def main():
    
    TOKEN = os.getenv('TELEGRAM_TOKEN')
    
    application = Application.builder().token(TOKEN).build()

    # Register the /start command handler
    application.add_handler(CommandHandler("start", start))

    # Register the message handler for text messages
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot
    application.run_polling()

if __name__ == '__main__':
    main()
