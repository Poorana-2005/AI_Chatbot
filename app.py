import streamlit as st
import ollama
import time
from PIL import Image, ImageEnhance
import pytesseract
import platform
import streamlit.components.v1 as components
import numpy as np
import cv2

# -------------------------
# Tesseract setup
# -------------------------
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="AI Bot", layout="wide")

# -------------------------
# CSS Styling
# -------------------------
st.markdown("""
<style>
.stApp { background-color: #E6E6FA; color: #2c2c2c; }

/* Sidebar */
.stSidebar { background-color: #f2f2f8 !important; }
.active-chat {
    background-color: #7a3db8 !important;
    color: white !important;
    border-radius: 8px;
    font-weight: bold;
    padding: 6px;
    margin-bottom: 4px;
}

/* Chat UI */
.user-bubble {
    background-color: #f0f0f0;
    padding: 10px 15px;
    border-radius: 12px;
    margin: 5px 0;
    max-width: 80%;
    text-align: right;
    margin-left: auto;
}
.ai-bubble {
    background-color: #d6b8ff;
    padding: 10px 15px;
    border-radius: 12px;
    margin: 5px 0;
    max-width: 80%;
    text-align: left;
}

/* Input Row */
.input-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 10px;
}
input[type="text"] {
    flex-grow: 1;
    padding: 10px;
    border-radius: 10px;
    border: 1px solid #ccc;
}

/* + Button */
.plus-btn {
    background-color: #7a3db8;
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 22px;
    padding: 6px 12px;
    cursor: pointer;
    transition: transform 0.1s ease-in-out, background-color 0.15s;
}
.plus-btn:hover {
    background-color: #9b63d9;
    transform: scale(1.15);
}

/* Hide uploader box */
[data-testid="stFileUploaderDropzone"] div div,
[data-testid="stFileUploaderDropzone"] section {
    display: none !important;
}
[data-testid="stFileUploader"] {
    height: 0 !important;
    overflow: hidden !important;
}

/* Small image preview */
.small-preview {
    max-width: 80px !important;
    max-height: 80px !important;
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 2px;
    background-color: white;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Session State
# -------------------------
if "all_chats" not in st.session_state:
    st.session_state["all_chats"] = {}
if "current_chat" not in st.session_state:
    st.session_state["current_chat"] = None
if "submitted_query" not in st.session_state:
    st.session_state["submitted_query"] = ""
if "ocr_submitted" not in st.session_state:
    st.session_state["ocr_submitted"] = False
if "uploaded_file_key" not in st.session_state:
    st.session_state["uploaded_file_key"] = 0
if "current_uploaded_file" not in st.session_state:
    st.session_state["current_uploaded_file"] = None

# -------------------------
# Ollama Model
# -------------------------
MODEL_NAME = "mistral:latest"

# -------------------------
# Sidebar
# -------------------------
st.sidebar.markdown("## 🤖 AI Chat")
if st.sidebar.button(" New Chat"):
    chat_id = str(len(st.session_state["all_chats"]) + 1)
    st.session_state["all_chats"][chat_id] = {"title": f"Chat {chat_id}", "messages": []}
    st.session_state["current_chat"] = chat_id
    st.session_state["ocr_submitted"] = False  # Reset for new chat
    st.session_state["uploaded_file_key"] += 1  # Reset uploader key
    st.session_state["current_uploaded_file"] = None  # Clear current uploaded file
    st.rerun()

st.sidebar.markdown("###  Chat History")
for chat_id, chat in st.session_state["all_chats"].items():
    btn_label = chat["title"]
    if chat_id == st.session_state["current_chat"]:
        st.sidebar.markdown(f"<div class='active-chat'>{btn_label}</div>", unsafe_allow_html=True)
    else:
        if st.sidebar.button(btn_label, key=f"chat_{chat_id}"):
            st.session_state["current_chat"] = chat_id
            st.session_state["ocr_submitted"] = False  # Reset OCR submission for selected chat
            st.session_state["uploaded_file_key"] += 1  # Reset uploader key
            st.session_state["current_uploaded_file"] = None  # Clear current uploaded file
            st.rerun()

# -------------------------
# Main Chat UI
# -------------------------
st.title("💬 AI Chatbot")

if st.session_state["current_chat"] is None:
    st.info("Start a new chat from the sidebar.")
else:
    chat = st.session_state["all_chats"][st.session_state["current_chat"]]

    # Display previous messages
    for msg in chat["messages"]:
        st.markdown(f"<div class='user-bubble'><b>You:</b> {msg['query']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='ai-bubble'><b>AI:</b> {msg['response']}</div>", unsafe_allow_html=True)

    # Input Row
    st.markdown('<div class="input-row">', unsafe_allow_html=True)
    query_input = st.text_input("", key="query_input", placeholder="Type message...", label_visibility="collapsed")
    
    # Use dynamic key for file uploader to allow multiple uploads
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], 
                                   key=f"hidden_uploader_{st.session_state['uploaded_file_key']}")

    # Inject + button
    components.html("""
        <script>
        const doc = window.parent.document;
        let inputBox = doc.querySelector('input[type="text"]');
        if (inputBox && !inputBox.parentElement.querySelector('.plus-btn')) {
            let plus = doc.createElement('button');
            plus.innerText = '🔗';
            plus.className = 'plus-btn';
            plus.onclick = () => {
                let uploader = doc.querySelector('input[type=file]');
                if (uploader) uploader.click();
            };
            inputBox.parentElement.appendChild(plus);
        }
        </script>
    """, height=0)
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------
    # OCR Processing (fully automatic)
    # -------------------------
    if uploaded_file is not None and uploaded_file != st.session_state["current_uploaded_file"]:
        try:
            # Clear previous uploaded file preview
            st.session_state["current_uploaded_file"] = uploaded_file
            
            image = Image.open(uploaded_file)
            
            # Create small preview (approximately 2cm square box)
            preview_size = 150  # 80px for small preview
            image.thumbnail((preview_size, preview_size))
            
            # Display small preview
            st.image(image, caption="Uploaded Image", use_container_width=False, width=preview_size, output_format="PNG")

            # Preprocess for OCR (use original image for OCR, not the thumbnail)
            image_for_ocr = Image.open(uploaded_file)
            image_for_ocr = image_for_ocr.convert("L")
            width, height = image_for_ocr.size
            if width < 1000:
                image_for_ocr = image_for_ocr.resize((width * 2, height * 2))
            enhancer = ImageEnhance.Contrast(image_for_ocr)
            image_for_ocr = enhancer.enhance(2)

            img_array = np.array(image_for_ocr)
            _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_for_ocr = Image.fromarray(img_array)

            # OCR
            extracted_text = pytesseract.image_to_string(image_for_ocr, lang='eng', config='--oem 3 --psm 6').strip()

            if extracted_text and not st.session_state["ocr_submitted"]:
                # Send OCR text to Ollama **only once**
                st.session_state.submitted_query = extracted_text
                st.session_state["ocr_submitted"] = True
            elif not extracted_text:
                st.warning("⚠️ No text detected in the image.")
        except Exception as e:
            st.error(f"⚠️ OCR failed: {e}")

    # -------------------------
    # Manual Query Send
    # -------------------------
    if st.button("Send", key="Send"):
        if query_input.strip():
            st.session_state.submitted_query = query_input
            # Reset uploaded file state when sending text query
            st.session_state["current_uploaded_file"] = None
            st.session_state["uploaded_file_key"] += 1  # Reset uploader for next use

    # -------------------------
    # AI Stream Response
    # -------------------------
    if st.session_state.submitted_query:
        query = st.session_state.submitted_query
        st.session_state.submitted_query = ""
        st.session_state["ocr_submitted"] = False  # Reset OCR submitted flag
        chat["messages"].append({"query": query, "response": ""})
        placeholder = st.empty()
        ai_text = ""

        try:
            # Build conversation history for context
            messages = []
            
            # Add previous messages to maintain context (last 6 messages for memory)
            for msg in chat["messages"][-6:]:  # Keep last 6 exchanges for context
                messages.append({"role": "user", "content": msg["query"]})
                messages.append({"role": "assistant", "content": msg["response"]})
            
            # Add the current query
            messages.append({"role": "user", "content": query})
            
            stream = ollama.chat(model=MODEL_NAME, messages=messages, stream=True)
            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    ai_text += chunk["message"]["content"]
                    placeholder.markdown(f"<div class='ai-bubble'><b>AI:</b> {ai_text}</div>", unsafe_allow_html=True)
                    time.sleep(0.01)
            
            chat["messages"][-1]["response"] = ai_text
        except Exception as e:
            chat["messages"][-1]["response"] = f"⚠️ Error: {e}"
            placeholder.markdown(f"<div class='ai-bubble'><b>AI:</b> ⚠️ Error: {e}</div>", unsafe_allow_html=True)
        
        chat["title"] = query[:25] + ("..." if len(query) > 25 else "")
        
        # Reset uploaded file state after processing
        st.session_state["current_uploaded_file"] = None
        st.session_state["uploaded_file_key"] += 1  # Reset uploader for next use
        st.rerun()