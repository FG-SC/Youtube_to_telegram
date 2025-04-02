import streamlit as st
from pytube import YouTube
import openai
import os
from dotenv import load_dotenv
import datetime
import googleapiclient.discovery
import tempfile
import whisper
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import re
from pytube.exceptions import PytubeError
import yt_dlp
from urllib.error import HTTPError
import requests
import logging
import subprocess
import shutil
import soundfile as sf
import asyncio
import nest_asyncio

# Patch the event loop
try:
    nest_asyncio.apply()
except:
    pass
    
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

# Load environment variables
load_dotenv()

# Initialize services
youtube_api_key = st.secrets["YOUTUBE_API_KEY"]
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=youtube_api_key)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load secret variables from Streamlit Cloud
TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHANNEL_ID = st.secrets["TELEGRAM_CHANNEL_ID"]
youtube
def verify_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

if not verify_ffmpeg():
    st.warning("FFmpeg is not properly installed. Audio processing may fail.")
# Update your Whisper model loading:
@st.cache_resource
def load_whisper_model():
    try:
        # Use the tiny or small model for production
        return whisper.load_model("tiny", device="cpu")  # or "small"
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
            data, samplerate = sf.read(audio_path)
            if len(data) == 0:
                return False, "Audio contains no data"
            return True, f"Valid audio file ({samplerate}Hz, {len(data)} samples)"
        except Exception as e:
            return False, f"Invalid audio format: {str(e)}"
            
    except Exception as e:
        return False, f"Validation error: {str(e)}"

# Update your download_youtube_audio function with these options:
def download_youtube_audio(url):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'force_generic_extractor': True,
            # Add these options:
            'cookiefile': 'cookies.txt',  # If you have YouTube cookies
            'referer': 'https://www.youtube.com',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'retries': 10,
            'fragment-retries': 10,
            'extractor-retries': 3,
            'socket-timeout': 30,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_path = ydl.prepare_filename(info).replace('.webm', '.mp3').replace('.m4a', '.mp3')
            return audio_path
    except Exception as e:
        logger.error(f"Download failed: {e}")
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
        logger.error(f"Audio file missing: {audio_path}")
        return None
    
    try:
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
            if len(audio_data) == 0:
                st.error("Audio file is empty")
                return None
                
        try:
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            
            _, probs = model.detect_language(mel)
            lang = max(probs, key=probs.get)
            logger.info(f"Detected language: {lang}")
            
            options = whisper.DecodingOptions(fp16=False)
            result = whisper.decode(model, mel, options)
            
            return clean_text(result.text)
            
        except RuntimeError as e:
            st.error(f"Whisper processing error: {e}")
            logger.error(f"Whisper RuntimeError: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        logger.error(f"Transcription error: {str(e)}")
        return None
    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)

def extract_video_id(url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

def get_video_details(video_id):
    try:
        request = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        )
        response = request.execute()

        if response["items"]:
            item = response["items"][0]
            snippet = item["snippet"]
            statistics = item["statistics"]
            return {
                "title": snippet["title"],
                "channel": snippet["channelTitle"],
                "published_at": snippet["publishedAt"],
                "views": int(statistics.get("viewCount", 0)),
                "likes": int(statistics.get("likeCount", 0)),
                "comments": int(statistics.get("commentCount", 0)),
                "description": snippet["description"],
                "thumbnail": snippet["thumbnails"]["high"]["url"]
            }
        return None
    except Exception as e:
        st.error(f"Error retrieving video details: {e}")
        return None

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

def create_pdf(video_details, transcription, summary):
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
        title = clean_text(video_details["title"])
        pdf.cell(200, 10, txt=title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(10)
        
        # Video details
        pdf.set_font(size=12, style="B")
        pdf.cell(200, 10, txt="Video Details", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font(size=12)
        
        details = f"""Channel: {clean_text(video_details["channel"])}
Published: {datetime.datetime.strptime(video_details["published_at"], '%Y-%m-%dT%H:%M:%SZ').strftime('%B %d, %Y')}
Views: {video_details["views"]:,}
Likes: {video_details["likes"]:,}
Comments: {video_details["comments"]:,}
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
            
            if response.status_code >= 200 and response.status_code < 300:
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
    video_id = extract_video_id(url)
    if not video_id:
        st.error("Invalid YouTube URL. Please enter a valid YouTube video URL.")
    
    with st.spinner("Fetching video details..."):
        video_details = get_video_details(video_id)
    
    if video_details:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(video_details["thumbnail"], width=200)
        with col2:
            st.subheader(video_details["title"])
            st.write(f"**Channel:** {video_details['channel']}")
            st.write(f"**Published:** {datetime.datetime.strptime(video_details['published_at'], '%Y-%m-%dT%H:%M:%SZ').strftime('%B %d, %Y')}")
            st.write(f"**Views:** {video_details['views']:,}")
            st.write(f"**Likes:** {video_details['likes']:,}")
            st.write(f"**Comments:** {video_details['comments']:,}")
        
        if st.button("Generate Transcription PDF"):
            with st.spinner("Downloading audio..."):
                audio_path = download_youtube_audio(url)
            
            if audio_path:
                st.success("Audio downloaded successfully!")
                
                # Validate audio file
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
                            with st.spinner("Creating PDF..."):
                                pdf_path = create_pdf(video_details, transcription, summary)
                            
                            if pdf_path:
                                with open(pdf_path, "rb") as f:
                                    st.download_button(
                                        label="Download PDF",
                                        data=f,
                                        file_name=f"{video_details['title']}_transcription.pdf",
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
