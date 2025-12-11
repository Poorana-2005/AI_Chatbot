# full_streamlit_chat_with_input_attachment.py
# Updated with persistent document icons in chat history and immediate message display

import streamlit as st
import ollama
import time
from PIL import Image, ImageEnhance
import pytesseract
import platform
import numpy as np
import cv2
import PyPDF2
import docx
from io import BytesIO
import base64
import io

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
# Load external CSS
# -------------------------
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# -------------------------
# Helper functions
# -------------------------
def image_bytes_to_base64(img_bytes: bytes, thumb_size=(120, 120)) -> str:
    """Return base64 PNG of a thumbnail from image bytes."""
    try:
        bio = BytesIO(img_bytes)
        img = Image.open(bio).convert("RGB")
        img.thumbnail(thumb_size)
        out = BytesIO()
        img.save(out, format="PNG")
        return base64.b64encode(out.getvalue()).decode()
    except Exception:
        return ""

def get_file_icon_svg(ext: str) -> str:
    """Return an SVG icon based on file extension."""
    ext = ext.lower()
    
    # PDF icon
    if ext in ["pdf"]:
        return '''<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="#d32f2f" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><path d="M9 13h6"></path><path d="M9 17h6"></path></svg>'''
    
    # Word/DOCX icon
    elif ext in ["doc", "docx"]:
        return '''<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="#1976d2" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>'''
    
    # Text/TXT icon
    elif ext in ["txt"]:
        return '''<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="#616161" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><line x1="10" y1="9" x2="8" y2="9"></line></svg>'''
    
    # Generic file icon
    else:
        return '''<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="#757575" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>'''

def make_chat_file_bubble_html(name: str, ext: str, img_b64: str = "") -> str:
    """Create the white file bubble HTML to insert into chat history when sending."""
    if img_b64:
        # Show image thumbnail for image files
        thumb_html = f"<img src='data:image/png;base64,{img_b64}' style='width:60px;height:60px;border-radius:8px;object-fit:cover;' />"
    else:
        # Show file icon for documents
        icon_svg = get_file_icon_svg(ext)
        thumb_html = f"<div style='width:60px;height:60px;border-radius:8px;background:#f9f9f9;display:flex;align-items:center;justify-content:center;'>{icon_svg}</div>"
    
    bubble = f"""
    <div style='display:flex; justify-content:flex-end; margin:10px 0;'>
      <div style='background:white; border-radius:14px; padding:10px; max-width:70%; box-shadow:0 1px 2px rgba(0,0,0,0.08);'>
        <div style='display:flex; gap:12px; align-items:center;'>
          {thumb_html}
          <div style='display:flex; flex-direction:column;'>
            <div style='font-weight:600; color:#222; font-size:14px;'>{name}</div>
            <div style='font-size:12px; color:#666; text-transform:uppercase;'>{ext}</div>
          </div>
        </div>
      </div>
    </div>
    """
    return bubble

# -------------------------
# Session State initialization
# -------------------------
if "all_chats" not in st.session_state:
    st.session_state["all_chats"] = {}
if "current_chat" not in st.session_state:
    st.session_state["current_chat"] = None
if "submitted_query" not in st.session_state:
    st.session_state["submitted_query"] = ""

# minimal chat structure
if "chat_store" not in st.session_state:
    st.session_state["chat_store"] = {}
if "chat_order" not in st.session_state:
    st.session_state["chat_order"] = []

# ✅ NEW: Add state management for two-step response generation
if "pending_response" not in st.session_state:
    st.session_state["pending_response"] = False
if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = ""

# -------------------------
# Ollama Model
# -------------------------
MODEL_NAME = "mistral:latest"

# -------------------------
# Document Processing Functions
# -------------------------
def extract_text_from_pdf_bytes(uploaded_file_bytes: bytes):
    try:
        bio = BytesIO(uploaded_file_bytes)
        pdf_reader = PyPDF2.PdfReader(bio)
        text_parts = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts).strip() if text_parts else None
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
        return None

def extract_text_from_docx_bytes(uploaded_file_bytes: bytes):
    try:
        bio = BytesIO(uploaded_file_bytes)
        doc = docx.Document(bio)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs).strip() if paragraphs else None
    except Exception as e:
        st.error(f"DOCX extraction error: {e}")
        return None

def extract_text_from_txt_bytes(uploaded_file_bytes: bytes):
    try:
        return uploaded_file_bytes.decode("utf-8", errors="replace").strip()
    except Exception as e:
        st.error(f"TXT extraction error: {e}")
        return None

def process_image_for_ocr_bytes(uploaded_file_bytes: bytes):
    try:
        bio = BytesIO(uploaded_file_bytes)
        image = Image.open(bio).convert("RGB")

        # Prepare for OCR using a grayscale copy
        image_for_ocr = image.convert("L")
        width, height = image_for_ocr.size
        if width < 1200:
            image_for_ocr = image_for_ocr.resize((width * 2, height * 2))
        enhancer = ImageEnhance.Contrast(image_for_ocr)
        image_for_ocr = enhancer.enhance(2.0)
        enhancer = ImageEnhance.Sharpness(image_for_ocr)
        image_for_ocr = enhancer.enhance(1.5)
        img_array = np.array(image_for_ocr)
        _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_for_ocr = Image.fromarray(img_array)

        extracted_text = pytesseract.image_to_string(
            image_for_ocr,
            lang='eng',
            config='--oem 3 --psm 6 -c preserve_interword_spaces=1'
        ).strip()

        return extracted_text if extracted_text else None
    except Exception as e:
        st.error(f"Image processing error: {e}")
        return None

def process_uploaded_file_bytes(uploaded_bytes: bytes, filename: str, mime: str = ""):
    """Decide which extractor to use given bytes and filename."""
    name = (filename or "").lower()
    mime = (mime or "").lower()
    if mime.startswith("image/") or any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
        return process_image_for_ocr_bytes(uploaded_bytes)
    if mime == "application/pdf" or name.endswith(".pdf"):
        return extract_text_from_pdf_bytes(uploaded_bytes)
    if "wordprocessingml" in mime or name.endswith(".docx"):
        return extract_text_from_docx_bytes(uploaded_bytes)
    if mime.startswith("text/") or name.endswith(".txt"):
        return extract_text_from_txt_bytes(uploaded_bytes)

    try:
        return uploaded_bytes.decode("utf-8", errors="replace")
    except Exception:
        st.warning("Unsupported file type for automatic text extraction.")
        return None

# -------------------------
# Sidebar (create/load chats)
# -------------------------
st.sidebar.markdown("## 🤖 AI Chat")
if st.sidebar.button("🆕 New Chat"):
    chat_id = str(len(st.session_state["chat_order"]) + 1)
    st.session_state["chat_store"][chat_id] = {
        "title": f"Chat {chat_id}",
        "messages": [],
        "documents": []
    }
    st.session_state["chat_order"].append(chat_id)
    st.session_state["current_chat"] = chat_id
    st.rerun()

st.sidebar.markdown("### 📜 Chat History")
for chat_id in st.session_state["chat_order"]:
    chat = st.session_state["chat_store"][chat_id]
    btn_label = chat["title"]
    if chat.get("documents"):
        btn_label = f"📄 {btn_label}"
    if chat_id == st.session_state.get("current_chat"):
        st.sidebar.markdown(f"<div class='active-chat'>{btn_label}</div>", unsafe_allow_html=True)
    else:
        if st.sidebar.button(btn_label, key=f"chat_{chat_id}"):
            st.session_state["current_chat"] = chat_id
            st.rerun()

# -------------------------
# Main Chat UI
# -------------------------
st.title("Public Policy Navigation")

if st.session_state.get("current_chat") is None:
    st.info("Start a new chat from the sidebar.")
    st.stop()

chat = st.session_state["chat_store"][st.session_state["current_chat"]]

# render messages
for msg in chat["messages"]:
    if msg["type"] == "file":
        # Always display file bubble from stored HTML
        st.markdown(msg["content"], unsafe_allow_html=True)
    elif msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='ai-bubble'><b>AI:</b> {msg['content']}</div>", unsafe_allow_html=True)

# -------------------------
# Chat Input with File Attachment Support
# -------------------------
st.markdown("<div class='input-container'>", unsafe_allow_html=True)

prompt = st.chat_input(
    "Say something and/or attach an image",
    accept_file=True,
    file_type=["jpg", "jpeg", "png", "pdf", "docx", "txt"],
)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Handle Chat Input Response
# -------------------------
if prompt:
    # Handle text message
    if prompt.text:
        display_text = prompt.text.strip()
        chat["messages"].append({"role": "user", "type": "text", "content": display_text})
    
    # Handle attached files
    combined_document_context = []
    
    if prompt["files"]:
        for uploaded_file in prompt["files"]:
            try:
                uploaded_bytes = uploaded_file.getvalue()
                fname = uploaded_file.name
                mime = uploaded_file.type or ""
                
                with st.spinner(f"🔎 Processing {fname}..."):
                    extracted_text = process_uploaded_file_bytes(uploaded_bytes, fname, mime)
                    
                    # Create image thumbnail or file icon
                    img_b64 = ""
                    is_image = False
                    
                    try:
                        if mime and mime.startswith("image/"):
                            img_b64 = image_bytes_to_base64(uploaded_bytes, thumb_size=(60,60))
                            is_image = True
                        else:
                            ext = fname.split('.')[-1].lower()
                            if ext in ["jpg", "jpeg", "png"]:
                                img_b64 = image_bytes_to_base64(uploaded_bytes, thumb_size=(60,60))
                                is_image = True
                    except Exception:
                        img_b64 = ""
                    
                    ext = (fname.split('.')[-1].upper() if fname else "FILE")
                    
                    # Create and store persistent file bubble HTML
                    file_bubble_html = make_chat_file_bubble_html(fname, ext, img_b64)
                    
                    # Store file message with metadata for persistent display
                    file_msg = {
                        "role": "user",
                        "type": "file",
                        "content": file_bubble_html,
                        "metadata": {
                            "filename": fname,
                            "extension": ext,
                            "is_image": is_image,
                            "thumbnail_b64": img_b64,
                            "has_text": bool(extracted_text)
                        }
                    }
                    chat["messages"].append(file_msg)
                    
                    # Collect extracted text
                    if extracted_text:
                        combined_document_context.append(f"--- Document: {fname} ---\n{extracted_text}")
                        
                        # Store document info
                        chat.setdefault("documents", []).append({
                            "name": fname,
                            "type": ext,
                            "text": extracted_text
                        })
                        st.success(f"✅ {fname} processed and attached")
                    else:
                        st.info(f"📎 {fname} attached (no text extracted)")
                        
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
    
    # Prepare query with all document contexts
    if combined_document_context:
        documents_text = "\n\n".join(combined_document_context)
        enhanced_query = f"""Based on these document contents:

{documents_text}

Please answer this question: {prompt.text if prompt.text else '[User attached documents]'} 

If the answer cannot be found in the document contents, please state that clearly."""
    else:
        enhanced_query = prompt.text if prompt.text else "[User attached files]"
    
    # ✅ Store query and trigger response generation on next run
    st.session_state["pending_query"] = enhanced_query
    st.session_state["pending_response"] = True
    st.rerun()  # Show user message first

# ✅ Generate AI response if pending
if st.session_state.get("pending_response", False):
    st.session_state["pending_response"] = False
    enhanced_query = st.session_state.get("pending_query", "")
    
    # -------------------------
    # AI stream response
    # -------------------------
    chat["messages"].append({"role": "assistant", "type": "text", "content": ""})
    ai_placeholder = st.empty()
    ai_text = ""

    try:
        messages = []
        system_prompt = """You are a helpful AI assistant. When users provide document content, carefully analyze it and provide accurate answers based on the provided text. For multiple documents, consider information from all documents. For long documents, pay attention to details across all sections and provide comprehensive answers. If the information isn't in the documents, clearly state that."""
        messages.append({"role": "system", "content": system_prompt})

        history = chat["messages"][-10:]
        for m in history:
            if m["type"] == "file":
                doc_names = ", ".join([d["name"] for d in chat.get("documents", [])])
                messages.append({"role": "user", "content": f"[User uploaded files: {doc_names}]"})
            elif m["role"] == "user" and m["type"] == "text":
                messages.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant" and m["type"] == "text" and m["content"]:
                messages.append({"role": "assistant", "content": m["content"]})

        messages.append({"role": "user", "content": enhanced_query})

        stream = ollama.chat(model=MODEL_NAME, messages=messages, stream=True)
        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                ai_text += chunk["message"]["content"]
                ai_placeholder.markdown(f"<div class='ai-bubble'><b>AI:</b> {ai_text}</div>", unsafe_allow_html=True)
                time.sleep(0.01)

        for i in range(len(chat["messages"]) - 1, -1, -1):
            if chat["messages"][i]["role"] == "assistant" and chat["messages"][i]["type"] == "text":
                chat["messages"][i]["content"] = ai_text
                break

    except Exception as e:
        error_msg = f"⚠️ Error: {e}"
        for i in range(len(chat["messages"]) - 1, -1, -1):
            if chat["messages"][i]["role"] == "assistant" and chat["messages"][i]["type"] == "text":
                chat["messages"][i]["content"] = error_msg
                break
        ai_placeholder.markdown(f"<div class='ai-bubble'><b>AI:</b> {error_msg}</div>", unsafe_allow_html=True)

    if len(chat["messages"]) > 0 and chat.get("title", "").startswith("Chat"):
        first_text = ""
        for m in chat["messages"]:
            if m["role"] == "user" and m["type"] == "text":
                first_text = m["content"]
                break
        if first_text:
            chat["title"] = first_text[:30] + ("..." if len(first_text) > 30 else "")

    st.rerun()
