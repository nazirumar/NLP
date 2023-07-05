import speech_recognition as sr

"""Problem
    You want to convert speech to text"""

r = sr.Recognizer()

with sr.Microphone() as source:
    print(".Please say something")
    audio = r.listen(source)
    print("Time over, thanks")
    try:
        print("I think you said: " + r.recognize_google(audio))
    except:
        pass