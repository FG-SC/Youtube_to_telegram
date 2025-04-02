import streamlit as st
import openai
import datetime
import tempfile
import whisper
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import re
import requests
import logging
import subprocess
import shutil
import os
import yt_dlp
import wave

# Disable unnecessary warnings
logging.getLogger("googleapiclient").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load secrets from Streamlit
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
    TELEGRAM_CHANNEL_ID = st.secrets["TELEGRAM_CHANNEL_ID"]
except Exception as e:
    st.error(f"Failed to load required secrets: {e}")
    st.stop()

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

def verify_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

if not verify_ffmpeg():
    st.warning("FFmpeg is not properly installed. Audio processing may fail.")

@st.cache_resource
def load_whisper_model():
    try:
        return whisper.load_model("small", device="cpu")
    except Exception as e:
        st.error(f"Failed to load Whisper model: {e}")
        return None

model = load_whisper_model()

def validate_audio_file(audio_path):
    try:
        if not os.path.exists(audio_path):
            return False, "File does not exist"
            
        file_size = os.path.getsize(audio_path)
        if file_size < 10 * 1024:
            return False, "File is too small (likely corrupted)"
        
        try:
            with wave.open(audio_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                if frames == 0:
                    return False, "Audio contains no data"
                return True, f"Valid WAV file ({wav_file.getframerate()}Hz)"
        except:
            try:
                with open(audio_path, 'rb') as f:
                    header = f.read(4)
                    if len(header) < 4:
                        return False, "File too small to be valid audio"
                    return True, "Non-WAV audio file"
            except:
                return False, "File is not readable"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def download_youtube_audio(url):
    try:
        temp_dir = tempfile.mkdtemp()
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(temp_dir, '%(id)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'force_generic_extractor': True,
            'referer': 'https://www.youtube.com',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'retries': 10,
            'fragment-retries': 10,
            'extractor-retries': 3,
            'socket-timeout': 30,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_path = ydl.prepare_filename(info).replace('.webm', '.wav').replace('.m4a', '.wav')
            
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not created at {audio_path}")
            
            if os.path.getsize(audio_path) < 10 * 1024:
                raise ValueError("Audio file is too small, likely corrupted")
            
            return audio_path
            
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return None

def clean_text(text):
    if not text:
        return ""
    
    replacements = {
        '’': "'", '‘': "'", '“': '"', '”': '"', '–': '-', '—': '-',
        '…': '...', '•': '*', '·': '*', '«': '"', '»': '"', '‹': "'", '›': "'",
    }
    
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    
    try:
        text.encode('ascii')
    except UnicodeEncodeError:
        text = text.encode('ascii', 'ignore').decode('ascii')
    
    return text.strip()

def transcribe_audio(audio_path):
    if not audio_path or not os.path.exists(audio_path):
        st.error(f"Audio file not found at: {audio_path}")
        return None
    
    try:
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
            if len(audio_data) == 0:
                st.error("Audio file is empty")
                return None
                
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        
        _, probs = model.detect_language(mel)
        lang = max(probs, key=probs.get)
        logger.info(f"Detected language: {lang}")
        
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)
        
        return clean_text(result.text)
        
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        logger.error(f"Transcription error: {str(e)}")
        return None
    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)

def generate_summary(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes video content."},
                {"role": "user", "content": f"Create a concise summary (100-200 words) of this video transcription:\n\n{text}"}
            ]
        )
        return clean_text(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

def create_pdf(video_title, channel, published_date, views, likes, comments, transcription, summary):
    try:
        pdf = FPDF()
        pdf.add_page()
        
        try:
            pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
            pdf.set_font("DejaVu", size=12)
        except:
            pdf.set_font("helvetica", size=12)
        
        # Title
        pdf.set_font(size=16, style="B")
        title = clean_text(video_title)
        pdf.cell(200, 10, txt=title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(10)
        
        # Video details
        pdf.set_font(size=12, style="B")
        pdf.cell(200, 10, txt="Video Details", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font(size=12)
        
        details = f"""Channel: {clean_text(channel)}
Published: {datetime.datetime.strptime(published_date, '%Y-%m-%dT%H:%M:%SZ').strftime('%B %d, %Y')}
Views: {views:,}
Likes: {likes:,}
Comments: {comments:,}
"""
        pdf.multi_cell(0, 10, txt=clean_text(details))
        pdf.ln(10)
        
        # Summary
        pdf.set_font(size=12, style="B")
        pdf.cell(200, 10, txt="Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font(size=12)
        pdf.multi_cell(0, 10, txt=clean_text(summary))
        pdf.ln(10)
        
        # Transcription
        pdf.set_font(size=12, style="B")
        pdf.cell(200, 10, txt="Full Transcription", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font(size=12)
        
        chunk_size = 1000
        transcription_chunks = [transcription[i:i+chunk_size] for i in range(0, len(transcription), chunk_size)]
        
        for chunk in transcription_chunks:
            pdf.multi_cell(0, 10, txt=clean_text(chunk))
            pdf.ln(5)
        
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(temp_pdf.name)
        return temp_pdf.name
        
    except Exception as e:
        st.error(f"Error creating PDF: {e}")
        logger.error(f"PDF creation error: {str(e)}")
        return None

def send_pdf_to_telegram(pdf_path, bot_token, chat_id):
    try:
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found at: {pdf_path}")
            return False
            
        file_size = os.path.getsize(pdf_path)
        if file_size == 0:
            logger.error("PDF file is empty")
            return False
            
        url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
        
        with open(pdf_path, 'rb') as f:
            files = {'document': f}
            data = {
                'chat_id': chat_id,
                'caption': 'YouTube Video Transcript'
            }
            
            response = requests.post(url, files=files, data=data, timeout=30)
            logger.info(f"Telegram API Response: {response.status_code}")
            
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Telegram API Error: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        logger.error(f"Error sending to Telegram: {str(e)}")
        return False

# Streamlit App
st.title("YouTube Video to PDF Transcriber")
st.write("Enter a YouTube video URL to generate a PDF with its transcription and details.")

url = st.text_input("YouTube Video URL:")

if url:
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    if st.button("Generate Transcription PDF"):
        with st.spinner("Downloading audio..."):
            audio_path = download_youtube_audio(url)
        
        if audio_path:
            st.success("Audio downloaded successfully!")
            
            is_valid, validation_msg = validate_audio_file(audio_path)
            if not is_valid:
                st.error(f"Invalid audio file: {validation_msg}")
            else:
                with st.spinner("Transcribing audio..."):
                    transcription = transcribe_audio(audio_path)
                
                if transcription:
                    with st.spinner("Generating summary..."):
                        summary = generate_summary(transcription)
                    
                    if summary:
                        # Get basic video info from URL
                        video_title = url.split('v=')[1][:11]  # Just use video ID as title
                        
                        with st.spinner("Creating PDF..."):
                            pdf_path = create_pdf(
                                video_title=video_title,
                                channel="YouTube Video",
                                published_date=datetime.datetime.now().isoformat(),
                                views=0,
                                likes=0,
                                comments=0,
                                transcription=transcription,
                                summary=summary
                            )
                        
                        if pdf_path:
                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    label="Download PDF",
                                    data=f,
                                    file_name=f"transcription_{video_title}.pdf",
                                    mime="application/pdf"
                                )
                            
                            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID:
                                if st.button("Send to Telegram"):
                                    with st.spinner("Sending to Telegram..."):
                                        if send_pdf_to_telegram(pdf_path, TELEGRAM_BOT_TOKEN, TELEGRAM_CHANNEL_ID):
                                            st.success("PDF sent to Telegram successfully!")
                                        else:
                                            st.error("Failed to send PDF to Telegram. Check logs for details.")
                            else:
                                st.warning("Telegram credentials not configured")
                            
                            try:
                                os.unlink(pdf_path)
                            except:
                                pass
                            
                            st.text_area("Transcription Preview", transcription, height=300)
                        else:
                            st.error("Failed to create PDF")
                    else:
                        st.error("Failed to generate summary")
                else:
                    st.error("Failed to transcribe audio")
        else:
            st.error("""
            Failed to download audio. Possible reasons:
            - Video is age-restricted or private
            - Network restrictions
            - YouTube rate limiting
            
            Try these solutions:
            1. Test with a different public video
            2. Wait and try again later
            """)
