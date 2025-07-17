import pyttsx3
import speech_recognition as sr
import threading

# Add a lock for thread safety
tts_lock = threading.Lock()

def initialize_text_to_speech():
    """Initialize the text-to-speech engine."""
    engine = pyttsx3.init()
    
    # Configure voice properties
    engine.setProperty('rate', 130)  # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
    
    # Get available voices and set to a preferred voice
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # Index 0 is usually the default male voice
    
    return engine

def speak_text():
    """Function to speak out the text."""
    with tts_lock:  # Use lock to prevent concurrent access
        try:
            engine = initialize_text_to_speech()
            text = "Fault is detected!.. Be safe"
            if text.strip():  # Only speak if there's actual text
                engine.say(text)
                engine.runAndWait()
        except RuntimeError as e:
            if "run loop already started" in str(e):
                # Just ignore this specific error
                pass
            else:
                raise

def thread_safe_speak():
    """A version that's safer to call from threads"""
    # Create a new process to handle the speech
    import multiprocessing
    process = multiprocessing.Process(target=speak_text)
    process.start()
    # Don't wait for it to complete

def main():
    """Main function to run the voice assistant."""
    speak_text()

if __name__ == "__main__":
    main()