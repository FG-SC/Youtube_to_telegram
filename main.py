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
load_dotenv()

# Initialize services with cache disabled
youtube = googleapiclient.discovery.build(
    "youtube", 
    "v3", 
    developerKey=os.getenv("YOUTUBE_API_KEY"),
    cache_discovery=False  # Disable cache to avoid warnings
)
openai.api_key = os.getenv("OPENAI_API_KEY")

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
        # Use base model for better accuracy while still being relatively fast
        return whisper.load_model("base")  # tiny, base, small, medium, large
    except Exception as e:
        st.error(f"Failed to load Whisper model: {e}")
        return None

model = load_whisper_model()

def clean_text(text):
    """Clean text for PDF generation by removing problematic characters"""
    if not text:
        return ""
    # Remove all non-ASCII characters
    return text.encode('ascii', 'ignore').decode('ascii').strip()

def download_youtube_audio(url):
    """Download YouTube audio with improved reliability and cookie support"""
    try:
        # Create temp directory for downloads
        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, "audio.wav")
        
        # Configure yt-dlp with proper headers and retries
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
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
            },
            'retries': 10,
            'fragment-retries': 10,
            'extractor-retries': 3,
            'socket-timeout': 30,
            'extract_flat': True,
            # Try to use cookies if available (helps with age-restricted content)
            'cookiefile': 'cookies.txt' if os.path.exists('cookies.txt') else None,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
                original_path = ydl.prepare_filename(info).replace('.webm', '.wav').replace('.m4a', '.wav')
                
                if not os.path.exists(original_path):
                    raise FileNotFoundError(f"Downloaded file not found at {original_path}")
                
                # Convert to proper WAV format
                converted_path = convert_to_wav(original_path, temp_audio_path)
                
                if not os.path.exists(converted_path):
                    raise FileNotFoundError(f"Converted file not found at {converted_path}")
                
                return converted_path
            except yt_dlp.utils.DownloadError as e:
                if "Private video" in str(e):
                    raise Exception("This video is private and cannot be accessed")
                elif "Sign in to confirm your age" in str(e):
                    raise Exception("Age-restricted video - please sign in to YouTube first")
                else:
                    raise e
                
    except Exception as e:
        logger.error(f"Download failed: {e}")
        # Clean up temp files if they exist
        try:
            if 'original_path' in locals() and os.path.exists(original_path):
                os.unlink(original_path)
            if 'converted_path' in locals() and os.path.exists(converted_path):
                os.unlink(converted_path)
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except:
            pass
        
        # Return the specific error message
        if "Age-restricted" in str(e):
            return None, "This video is age-restricted. Please sign in to YouTube in your browser first."
        elif "Private video" in str(e):
            return None, "This video is private and cannot be accessed."
        else:
            return None, "Failed to download video. YouTube may be temporarily blocking our requests. Please try again later."

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

            
    with st.spinner("Fetching video details..."):
        video_details = get_video_details(video_id)
    
    if not video_details:
        st.error("Failed to fetch video details. The video may be private or unavailable.")

                
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
