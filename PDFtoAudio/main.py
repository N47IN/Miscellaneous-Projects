import pyttsx3,PyPDF2
from gtts import gTTS
#insert name of your pdf 
pdfreader = PyPDF2.PdfReader(open('Navin_CV.pdf', 'rb'))
speaker = pyttsx3.init()

for page_num in range(len(pdfreader.pages)):
    text = pdfreader.pages[page_num].extract_text()
    clean_text = text.strip().replace('\n', ' ')
    print(clean_text)
#name mp3 file whatever you would like
speech = gTTS(text = clean_text)
speech.save("audio.mp3")

speaker.stop()
