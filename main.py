from transformers import pipeline
import streamlit as st
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" #tesseract fix


from transformers import pipeline

pipe = pipeline("document-question-answering", model="impira/layoutlm-document-qa")

st.title("Document-Question-Answering using LayoutLm")
image_ = st.file_uploader("Choose an image to ask about",['jpg','png','jpeg'])
question = st.text_input("ask your question")   

if image_ and question:
    image = Image.open(image_)
    st.image(image, caption="Uploaded Document", use_container_width=True)
    with st.spinner("finding answer..."):
        outputs = pipe(image, question=question)
    for output in outputs:
    # print(f"Answer: {output['answer']}")
        st.write(f'Answer: {output['answer']}')
    
#streamlit run main.py
# .venv/Scripts/activate