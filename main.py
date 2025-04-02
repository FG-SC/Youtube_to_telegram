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
import logging
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize services
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))
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
        return whisper.load_model("base")  # You can use "small", "medium", or "large" for better quality
    except Exception as e:
        st.error(f"Failed to load Whisper model: {e}")
        return None

model = load_whisper_model()

def clean_text(text):
    """
    Clean text for PDF generation by:
    - Replacing problematic Unicode characters with ASCII equivalents
    - Removing unsupported emojis (with option to keep some)
    - Ensuring text is PDF-compatible
    """
    if not text:
        return ""
    
    # Standard replacements
    replacements = {
        '’': "'", '‘': "'", '“': '"', '”': '"', '–': '-', '—': '-',
        '…': '...', '•': '*', '·': '*', '«': '"', '»': '"', '‹': "'", '›': "'",
        '™': '(TM)', '®': '(R)', '©': '(C)', '±': '+/-', 'µ': 'u', '°': ' deg',
    }
    
    # First pass - standard replacements
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    
    # Second pass - handle emojis by removing them (or replace with text if preferred)
    # Remove all emojis and other non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    return text.strip()

def download_youtube_audio(url):
    """Download YouTube audio with improved error handling and retries"""
    try:
        # Create temp directory for downloads
        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, "audio.wav")
        
        # Try yt-dlp first with proper headers to avoid 403 errors
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
            },
            'retries': 3,
            'fragment-retries': 3,
            'skip-unavailable-fragments': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            original_path = ydl.prepare_filename(info).replace('.webm', '.wav').replace('.m4a', '.wav')
            
            if not os.path.exists(original_path):
                raise FileNotFoundError(f"Downloaded file not found at {original_path}")
            
            # Convert to proper WAV format if needed
            converted_path = convert_to_wav(original_path, temp_audio_path)
            
            if not os.path.exists(converted_path):
                raise FileNotFoundError(f"Converted file not found at {converted_path}")
            
            return converted_path
            
    except Exception as e:
        logger.error(f"Download with yt-dlp failed: {e}")
        # Fallback to pytube if yt-dlp fails
        try:
            return download_with_pytube(url)
        except Exception as e:
            st.error(f"Failed to download audio: {e}")
            return None

def download_with_pytube(url):
    """Fallback download function using pytube"""
    try:
        yt = YouTube(
            url,
            use_oauth=True,
            allow_oauth_cache=True,
            on_progress_callback=lambda stream, chunk, bytes_remaining: st.write("Downloading...")
        )
        
        audio_stream = yt.streams.filter(only_audio=True).order_by('abr').last()
        
        if not audio_stream:
            raise PytubeError("No audio stream available")
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            audio_stream.download(filename=tmp_file.name)
            return convert_to_wav(tmp_file.name)
            
    except Exception as e:
        logger.error(f"Pytube download failed: {e}")
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
        if input_path != output_path:
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
        result = model.transcribe(audio_path)
        return clean_text(result["text"])
    except Exception as e:
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
                "thumbnail": snippet["thumbnails"]["high"]["url"]
            }
        return None
    except Exception as e:
        st.error(f"Error retrieving video details: {e}")
        return None

def generate_summary(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes video content."},
                {"role": "user", "content": f"Create a concise summary (about 500 words) of the following video transcription if it has more than a thousand words, if not, make the summary about 100 words:\n\n{text}"}
            ]
        )
        return clean_text(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

def create_pdf(video_details, transcription, summary):
    pdf = FPDF()
    pdf.add_page()
    
    # Set font - using standard Helvetica and cleaning text to avoid Unicode issues
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

# Streamlit app
st.title("YouTube Video to PDF Transcriber")
st.write("Enter a YouTube video URL to generate a PDF with its transcription and details.")

url = st.text_input("YouTube Video URL:")

if url:
    video_id = extract_video_id(url)
    if not video_id:
        st.error("Invalid YouTube URL. Please enter a valid YouTube video URL.")
        return
    
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
                st.error("""
                Failed to download audio. Possible reasons:
                - Video is age-restricted (try signing in to YouTube in your browser first)
                - Video is not available in your region
                - Video is private or removed
                - Network restrictions
                - YouTube is rate limiting our requests (try again later)
                """)
