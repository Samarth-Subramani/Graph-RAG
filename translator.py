from langdetect import detect
from deep_translator import GoogleTranslator

CHUNK_SIZE = 4500

def translate_to_english(text, chunk_size=CHUNK_SIZE):
    if not text.strip():
        return ""
    try:
        detected_lang = detect(text)
        if detected_lang == "en":
            return text

        translated_chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            translated = GoogleTranslator(source='auto', target='en').translate(chunk)
            translated_chunks.append(translated)

        return ' '.join(translated_chunks)

    except Exception as e:
        print(f"Translation error: {e}")
        return text
