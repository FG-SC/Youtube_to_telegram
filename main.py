import streamlit as st
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
import yt_dlp
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
# Load environment variables
load_dotenv()

# Gracefully handle missing environment variables
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    st.error("YouTube API key is missing. Please set it in your environment variables.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key is missing. Please set it in your environment variables.")
else:
    openai.api_key = OPENAI_API_KEY

# Only initialize YouTube API if we have a key
youtube = None
if YOUTUBE_API_KEY:
    try:
        youtube = googleapiclient.discovery.build(
            "youtube", 
            "v3", 
            developerKey=YOUTUBE_API_KEY,
            cache_discovery=False
        )
    except Exception as e:
        st.error(f"Failed to initialize YouTube API: {e}")
        
# Verify FFmpeg installation
def verify_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

if not verify_ffmpeg():
    st.warning("FFmpeg is not properly installed. Some audio processing may fail.")
    
# Initialize Whisper model with error handling
@st.cache_resource
def load_whisper_model():
    try:
        # Use tiny model for faster transcription and lower memory usage
        return whisper.load_model("tiny")
    except Exception as e:
        st.error(f"Failed to load Whisper model: {e}")
        logger.error(f"Whisper model error: {str(e)}")
        return None

# Initialize model only when needed
def get_whisper_model():
    model = load_whisper_model()
    if model is None:
        st.error("Failed to initialize speech recognition model. Please try again later.")
    return model
model = load_whisper_model()

def clean_text(text):
    """Clean text for PDF generation by removing problematic characters"""
    if not text:
        return ""
    # Remove all non-ASCII characters
    return text.encode('ascii', 'ignore').decode('ascii').strip()

def download_youtube_audio(url):
    """Download YouTube audio with improved reliability"""
    try:
        # Create temp directory for downloads
        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, "audio.wav")
        
        # Configure yt-dlp with additional options for reliability
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
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
            },
            'retries': 30,
            'fragment-retries': 30,  
            'extractor-retries': 10,
            'socket-timeout': 60,
            'extract_flat': True,
            'external_downloader': 'native',  # Don't use external downloaders
            'nocheckcertificate': True,  # Skip SSL verification
            'geo_bypass': True,  # Try to bypass geo-restrictions
            'ignoreerrors': True,  # Continue on download errors
        }
        
        # Try multiple download attempts
        for attempt in range(3):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    if not info:
                        continue
                        
                    original_path = ydl.prepare_filename(info).replace('.webm', '.wav').replace('.m4a', '.wav')
                    
                    if os.path.exists(original_path):
                        return original_path
                    
                    st.warning(f"Download attempt {attempt+1} failed. Retrying...")
                    time.sleep(3)  # Wait before retry
            except Exception as e:
                logger.error(f"Download attempt {attempt+1} failed: {e}")
                time.sleep(5)  # Longer wait after exception
        
        # If all attempts fail, try direct ffmpeg download
        st.warning("Trying alternative download method...")
        try:
            # Get video ID for direct stream attempt
            video_id = url.split("v=")[1].split("&")[0]
            direct_url = f"https://www.youtube.com/watch?v={video_id}"
            
            # Direct FFmpeg download
            subprocess.run([
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
                "-i", direct_url,
                "-vn",
                "-ar", "16000",
                "-ac", "1",
                temp_audio_path
            ], check=True, timeout=300)
            
            if os.path.exists(temp_audio_path):
                return temp_audio_path
        except Exception as e:
            logger.error(f"Alternative download failed: {e}")
        
        return None
            
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None
        
def convert_to_wav(input_path, output_path=None):
    """Convert audio file to WAV format using FFmpeg"""
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    
    try:
        subprocess.run([
            "ffmpeg",
            "-i", input_path,
            "-ac", "1",  # Mono audio
            "-ar", "16000",  # 16kHz sample rate
            "-y",  # Overwrite without asking
            output_path
        ], check=True)
        # Clean up the original file
        if input_path != output_path and os.path.exists(input_path):
            os.unlink(input_path)
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e}")
        return None

def transcribe_audio(audio_path):
    if not audio_path or not os.path.exists(audio_path):
        st.error(f"Audio file not found at: {audio_path}")
        return None
    
    try:
        # Show progress while transcribing
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(progress):
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Transcribing... {int(progress*100)}% complete")
            return True
        
        # Use ThreadPoolExecutor to run transcription in background
        with ThreadPoolExecutor() as executor:
            future = executor.submit(model.transcribe, audio_path, progress_callback=progress_callback)
            result = future.result()
        
        progress_bar.empty()
        status_text.empty()
        return clean_text(result["text"])
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Error transcribing audio: {e}")
        logger.error(f"Transcription error: {str(e)}")
        return None

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
                "thumbnail": snippet["thumbnails"]["high"]["url"],
                "video_id": video_id
            }
        return None
    except Exception as e:
        st.error(f"Error retrieving video details: {e}")
        return None

def generate_summary(text):
    try:
        # Only generate summary if text is long enough
        if len(text.split()) < 100:
            return "Video content too short for summary"
            
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Using faster model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes video content."},
                {"role": "user", "content": f"Create a concise summary (about 100 words) of this video:\n\n{text[:3000]}"}  # Limit input size
            ],
            max_tokens=150
        )
        return clean_text(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

def create_pdf(video_details, transcription, summary):
    pdf = FPDF()
    pdf.add_page()
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
    if summary:
        pdf.set_font(size=12, style="B")
        pdf.cell(200, 10, txt="Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font(size=12)
        pdf.multi_cell(0, 10, txt=clean_text(summary))
        pdf.ln(10)
    
    # Transcription
    pdf.set_font(size=12, style="B")
    pdf.cell(200, 10, txt="Full Transcription", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(size=12)
    
    # Split transcription into chunks to avoid memory issues
    chunk_size = 1000
    transcription_chunks = [transcription[i:i+chunk_size] for i in range(0, len(transcription), chunk_size)]
    
    for chunk in transcription_chunks:
        pdf.multi_cell(0, 10, txt=clean_text(chunk))
        pdf.ln(5)
    
    # Save to temporary file
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_pdf.name)
    return temp_pdf.name

# Main Streamlit app
st.title("YouTube Video to PDF Transcriber")
st.write("Enter a YouTube video URL to generate a PDF with its transcription.")

url = st.text_input("YouTube Video URL:")

if url:
    video_id = extract_video_id(url)
    if not video_id:
        st.error("Invalid YouTube URL. Please enter a valid YouTube video URL.")
    elif not youtube:
        st.error("YouTube API not initialized. Please check your API key.")
    else:
        with st.spinner("Fetching video details..."):
            try:
                video_details = get_video_details(video_id)
                if not video_details:
                    st.error("Failed to fetch video details. The video may be private or unavailable.")
            except Exception as e:
                st.error(f"Error fetching video details: {str(e)}")
                video_details = None
        
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
        with st.spinner("Downloading audio (this may take a few minutes)..."):
            audio_path, error_msg = download_youtube_audio(url)
        
        if audio_path:
            with st.spinner("Transcribing audio..."):
                transcription = transcribe_audio(audio_path)
                try:
                    os.unlink(audio_path)
                except:
                    pass
            
            if transcription:
                with st.spinner("Generating summary..."):
                    summary = generate_summary(transcription)
                
                if summary:
                    with st.spinner("Creating PDF..."):
                        pdf_path = create_pdf(video_details, transcription, summary)
                    
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="Download PDF",
                            data=f,
                            file_name=f"{video_details['title']}_transcription.pdf",
                            mime="application/pdf"
                        )
                    
                    try:
                        os.unlink(pdf_path)
                    except:
                        pass
                    
                    st.success("PDF generated successfully!")
                    st.text_area("Transcription Preview", transcription, height=300)
                else:
                    st.error("Failed to generate summary.")
            else:
                st.error("Failed to transcribe audio.")
        else:
            st.error(error_msg if error_msg else """
            Failed to download audio. Possible reasons:
            - Video is age-restricted (try signing in to YouTube in your browser first)
            - Video is not available in your region
            - Video is private or removed
            - Network restrictions
            - YouTube is rate limiting our requests (try again later)
            """)
