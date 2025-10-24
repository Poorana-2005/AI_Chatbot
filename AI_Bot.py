import streamlit as st
import ollama
import time
from PIL import Image, ImageEnhance
import pytesseract
import platform
import streamlit.components.v1 as components
import numpy as np
import cv2
import PyPDF2
import docx
from io import BytesIO

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

/* Document info box */
.document-info {
    background-color: #f8f5ff;
    border: 1px solid #d6b8ff;
    border-radius: 8px;
    padding: 10px;
    margin: 10px 0;
    font-size: 0.9em;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Document Processing Functions
# -------------------------

def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF files including multi-page documents"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"{page_text}\n"
        return text.strip()
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
        return None

def extract_text_from_docx(uploaded_file):
    """Extract text from DOCX files"""
    try:
        doc = docx.Document(uploaded_file)
        text = ""
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"DOCX extraction error: {e}")
        return None

def extract_text_from_txt(uploaded_file):
    """Extract text from TXT files"""
    try:
        return uploaded_file.getvalue().decode("utf-8").strip()
    except Exception as e:
        st.error(f"TXT extraction error: {e}")
        return None

def process_image_for_ocr(uploaded_file):
    """Process image files using OCR with enhanced preprocessing"""
    try:
        image = Image.open(uploaded_file)
        
        # Create small preview
        preview_size = 150
        image.thumbnail((preview_size, preview_size))
        st.image(image, caption="Uploaded Image", use_container_width=False, width=preview_size, output_format="PNG")

        # Enhanced preprocessing for OCR
        image_for_ocr = Image.open(uploaded_file)
        image_for_ocr = image_for_ocr.convert("L")
        
        # Resize for better OCR accuracy on small text
        width, height = image_for_ocr.size
        if width < 1200:
            image_for_ocr = image_for_ocr.resize((width * 2, height * 2))
        
        # Enhance image for better text recognition
        enhancer = ImageEnhance.Contrast(image_for_ocr)
        image_for_ocr = enhancer.enhance(2.0)
        enhancer = ImageEnhance.Sharpness(image_for_ocr)
        image_for_ocr = enhancer.enhance(1.5)

        # Apply thresholding
        img_array = np.array(image_for_ocr)
        _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_for_ocr = Image.fromarray(img_array)

        # Enhanced OCR configuration for better accuracy
        extracted_text = pytesseract.image_to_string(
            image_for_ocr, 
            lang='eng', 
            config='--oem 3 --psm 6 -c preserve_interword_spaces=1'
        ).strip()
        
        return extracted_text
    except Exception as e:
        st.error(f"Image processing error: {e}")
        return None

def process_uploaded_file(uploaded_file):
    """Main function to process any uploaded file type"""
    file_type = uploaded_file.type
    extracted_text = ""
    
    if file_type in ["image/png", "image/jpeg", "image/jpg"]:
        extracted_text = process_image_for_ocr(uploaded_file)
        
    elif file_type == "application/pdf":
        extracted_text = extract_text_from_pdf(uploaded_file)
        
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        extracted_text = extract_text_from_docx(uploaded_file)
        
    elif file_type == "text/plain":
        extracted_text = extract_text_from_txt(uploaded_file)
    
    return extracted_text

# -------------------------
# Session State
# -------------------------
if "all_chats" not in st.session_state:
    st.session_state["all_chats"] = {}
if "current_chat" not in st.session_state:
    st.session_state["current_chat"] = None
if "submitted_query" not in st.session_state:
    st.session_state["submitted_query"] = ""
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
if st.sidebar.button("🆕 New Chat"):
    chat_id = str(len(st.session_state["all_chats"]) + 1)
    st.session_state["all_chats"][chat_id] = {
        "title": f"Chat {chat_id}", 
        "messages": [],
        "document_context": "",
        "document_name": "",
        "document_type": ""
    }
    st.session_state["current_chat"] = chat_id
    st.session_state["uploaded_file_key"] += 1
    st.session_state["current_uploaded_file"] = None
    st.rerun()

st.sidebar.markdown("### 📜 Chat History")
for chat_id, chat in st.session_state["all_chats"].items():
    btn_label = chat["title"]
    # Add document indicator to chat title if there's a document
    if chat.get("document_name"):
        btn_label = f"📄 {btn_label}"
    
    if chat_id == st.session_state["current_chat"]:
        st.sidebar.markdown(f"<div class='active-chat'>{btn_label}</div>", unsafe_allow_html=True)
    else:
        if st.sidebar.button(btn_label, key=f"chat_{chat_id}"):
            st.session_state["current_chat"] = chat_id
            st.session_state["uploaded_file_key"] += 1
            st.session_state["current_uploaded_file"] = None
            st.rerun()

# -------------------------
# Main Chat UI
# -------------------------
st.title("💬 AI Chatbot")

if st.session_state["current_chat"] is None:
    st.info("Start a new chat from the sidebar.")
else:
    chat = st.session_state["all_chats"][st.session_state["current_chat"]]

    # Display document info if available (but not the full text)
    if chat.get("document_name"):
        st.markdown(f"""
        <div class="document-info">
            <strong>📄 Active Document:</strong> {chat["document_name"]} ({chat["document_type"]})<br>
            <small>Document is available for questioning. Ask about its content.</small>
        </div>
        """, unsafe_allow_html=True)

    # Display previous messages
    for msg in chat["messages"]:
        st.markdown(f"<div class='user-bubble'><b>You:</b> {msg['query']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='ai-bubble'><b>AI:</b> {msg['response']}</div>", unsafe_allow_html=True)

    # Input Row
    st.markdown('<div class="input-row">', unsafe_allow_html=True)
    query_input = st.text_input("", key="query_input", placeholder="Type message...", label_visibility="collapsed")
    
    # Use dynamic key for file uploader to allow multiple file types
    uploaded_file = st.file_uploader("", 
                                   type=["png", "jpg", "jpeg", "pdf", "docx", "txt"], 
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
    # File Processing (all file types) - SILENT MODE
    # -------------------------
    if uploaded_file is not None and uploaded_file != st.session_state["current_uploaded_file"]:
        try:
            # Clear previous uploaded file
            st.session_state["current_uploaded_file"] = uploaded_file
            
            # Show processing message
            with st.spinner(f"📄 Processing {uploaded_file.name}..."):
                # Process the file based on its type
                extracted_text = process_uploaded_file(uploaded_file)
                
                if extracted_text:
                    # Store the document context in the chat session (not displayed in query)
                    chat["document_context"] = extracted_text
                    chat["document_name"] = uploaded_file.name
                    chat["document_type"] = uploaded_file.type.split('/')[-1].upper()
                    
                    # Show success message
                    st.success(f"✅ {uploaded_file.name} uploaded successfully!")
                    
                    # Show document stats
                    word_count = len(extracted_text.split())
                    char_count = len(extracted_text)
                    st.caption(f"📊 Document ready: {word_count} words, {char_count} characters")
                    
                    # Don't display the extracted text in the chat
                    # It's stored in chat["document_context"] for future queries
                    
                else:
                    st.warning("⚠️ No text could be extracted from the file.")
                    
        except Exception as e:
            st.error(f"⚠️ File processing failed: {e}")

    # -------------------------
    # Manual Query Send
    # -------------------------
    if st.button("Send", key="Send"):
        if query_input.strip():
            # Store only the user's visible query for display
            display_query = query_input
            
            # If we have document context in this chat, create enhanced query but don't show it
            if chat.get("document_context"):
                # Create enhanced query with document context but don't display it
                enhanced_query = f"""Based on this document content:
{chat["document_context"]}

Please answer this question: {query_input}

If the answer cannot be found in the document content, please state that clearly."""
            else:
                enhanced_query = query_input
            
            # Set the enhanced query for processing but display only user's original query
            st.session_state.submitted_query = enhanced_query
            
            # Add only the user's visible query to chat history
            chat["messages"].append({"query": display_query, "response": ""})
            
            # Reset uploaded file state when sending text query
            st.session_state["current_uploaded_file"] = None
            st.session_state["uploaded_file_key"] += 1

    # -------------------------
    # AI Stream Response
    # -------------------------
    if st.session_state.submitted_query:
        query = st.session_state.submitted_query
        st.session_state.submitted_query = ""
        
        # Find the last message (which should be the one we just added) and update response
        if chat["messages"]:
            placeholder = st.empty()
            ai_text = ""

            try:
                # Build conversation history for context
                messages = []
                
                # Add system message for better document understanding
                system_prompt = """You are a helpful AI assistant. When users provide document content, carefully analyze it and provide accurate answers based on the provided text. For long documents, pay attention to details across all sections and provide comprehensive answers. If the information isn't in the document, clearly state that."""
                messages.append({"role": "system", "content": system_prompt})
                
                # Add previous messages to maintain context (last 6 messages for memory)
                # Skip the current message since we're processing it
                for msg in chat["messages"][-7:-1]:  # Get previous 6 messages
                    messages.append({"role": "user", "content": msg['query']})
                    messages.append({"role": "assistant", "content": msg['response']})
                
                # Add the current enhanced query (with document context if available)
                messages.append({"role": "user", "content": query})
                
                # Get response from Ollama
                stream = ollama.chat(model=MODEL_NAME, messages=messages, stream=True)
                for chunk in stream:
                    if "message" in chunk and "content" in chunk["message"]:
                        ai_text += chunk["message"]["content"]
                        placeholder.markdown(f"<div class='ai-bubble'><b>AI:</b> {ai_text}</div>", unsafe_allow_html=True)
                        time.sleep(0.01)
                
                # Update the response for the last message
                chat["messages"][-1]["response"] = ai_text
                
            except Exception as e:
                error_msg = f"⚠️ Error: {e}"
                chat["messages"][-1]["response"] = error_msg
                placeholder.markdown(f"<div class='ai-bubble'><b>AI:</b> {error_msg}</div>", unsafe_allow_html=True)
            
            # Update chat title with first meaningful query
            if len(chat["messages"]) == 1:
                first_query = chat["messages"][0]["query"][:30]
                chat["title"] = first_query + ("..." if len(first_query) > 30 else "")
        
        # Reset uploaded file state after processing
        st.session_state["current_uploaded_file"] = None
        st.session_state["uploaded_file_key"] += 1
        st.rerun()