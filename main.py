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
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Gracefully handle missing environment variables
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    st.error("YouTube API key is missing. Please add it to your .env file or Streamlit secrets.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key is missing. Please add it to your .env file or Streamlit secrets.")
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

ffmpeg_available = verify_ffmpeg()
if not ffmpeg_available:
    st.warning("FFmpeg is not properly installed. Audio processing will fail.")
    
# Initialize Whisper model with error handling
@st.cache_resource
def load_whisper_model():
    try:
        # Use tiny model for faster transcription and lower memory usage
        # This is crucial for Streamlit Cloud deployment
        return whisper.load_model("tiny")
    except Exception as e:
        st.error(f"Failed to load Whisper model: {e}")
        logger.error(f"Whisper model error: {str(e)}")
        return None

# Load model only once
model = load_whisper_model()

def clean_text(text):
    """Clean text for PDF generation by removing problematic characters"""
    if not text:
        return ""
    # Remove all non-ASCII characters for maximum compatibility
    return text.encode('ascii', 'ignore').decode('ascii').strip()

def convert_to_wav(input_path, output_path=None):
    """Convert audio file to WAV format using FFmpeg"""
    if not ffmpeg_available:
        return None
        
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
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Clean up the original file
        if input_path != output_path and os.path.exists(input_path):
            try:
                os.unlink(input_path)
            except:
                pass
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e}")
        return None

def download_youtube_audio(url):
    """Download YouTube audio with improved reliability"""
    try:
        # Create temp directory for downloads
        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, "audio.wav")
        
        # Configure yt-dlp with options for reliability
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
            'retries': 5,
            'socket-timeout': 30,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            original_path = ydl.prepare_filename(info).replace('.webm', '.wav').replace('.m4a', '.wav')
            
            if not os.path.exists(original_path):
                return None, f"Downloaded file not found at {original_path}"
            
            # Convert to proper WAV format
            converted_path = convert_to_wav(original_path, temp_audio_path)
            
            if not converted_path or not os.path.exists(converted_path):
                return None, f"Audio conversion failed"
            
            return converted_path, None
            
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None, f"Download failed: {str(e)}"

def transcribe_audio(audio_path):
    """Transcribe audio file using Whisper model"""
    if not audio_path or not os.path.exists(audio_path):
        return None, "Audio file not found"
    
    if model is None:
        return None, "Whisper model failed to load"
    
    try:
        # Show progress indicator
        progress_placeholder = st.empty()
        progress_placeholder.text("Transcribing audio... This may take a few minutes.")
        
        # Perform transcription
        result = model.transcribe(audio_path)
        
        # Clear progress indicator
        progress_placeholder.empty()
        
        # Clean up temp file
        try:
            os.unlink(audio_path)
        except:
            pass
            
        return clean_text(result["text"]), None
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return None, f"Transcription failed: {str(e)}"

def extract_video_id(url):
    """Extract YouTube video ID from URL"""
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

def get_video_details(video_id):
    """Get video details from YouTube API"""
    if not youtube:
        return None, "YouTube API not initialized"
        
    try:
        request = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        )
        response = request.execute()

        if not response.get("items"):
            return None, "Video not found or may be private"
            
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
        }, None
    except Exception as e:
        logger.error(f"Error retrieving video details: {e}")
        return None, f"Failed to get video details: {str(e)}"

def generate_summary(text):
    """Generate summary using OpenAI API"""
    if not OPENAI_API_KEY:
        return "OpenAI API key not configured", None
        
    try:
        # Only generate summary if text is long enough
        if len(text.split()) < 100:
            return "Video content too short for summary", None
            
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes video content."},
                {"role": "user", "content": f"Create a concise summary (about 100 words) of this video transcript if it has less than a thousand words. If it has more than that, create a summary with 500 words:\n\n{text[:4000]}"}
            ],
            max_tokens=200
        )
        return clean_text(response.choices[0].message.content), None
    except Exception as e:
        logger.error(f"Summary generation error: {str(e)}")
        return None, f"Failed to generate summary: {str(e)}"

def create_pdf(video_details, transcription, summary=None):
    """Create PDF with video details, summary, and transcription"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", size=12)
        
        # Title
        pdf.set_font(size=16, style="B")
        title = clean_text(video_details["title"])
        pdf.cell(200, 10, txt=title[:70], new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(10)
        
        # Video details
        pdf.set_font(size=12, style="B")
        pdf.cell(200, 10, txt="Video Details", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font(size=12)
        
        # Format published date
        try:
            published_date = datetime.datetime.strptime(
                video_details["published_at"], 
                '%Y-%m-%dT%H:%M:%SZ'
            ).strftime('%B %d, %Y')
        except:
            published_date = video_details["published_at"]
        
        details = f"""Channel: {clean_text(video_details["channel"])}
Published: {published_date}
Views: {video_details["views"]:,}
Likes: {video_details["likes"]:,}
Comments: {video_details["comments"]:,}
"""
        pdf.multi_cell(0, 10, txt=clean_text(details))
        pdf.ln(10)
        
        # Summary (if available)
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
        pdf_path = temp_pdf.name
        pdf.output(pdf_path)
        return pdf_path, None
    except Exception as e:
        logger.error(f"PDF creation error: {str(e)}")
        return None, f"Failed to create PDF: {str(e)}"

# Main Streamlit app
st.title("YouTube Video to PDF Transcriber")
st.write("Enter a YouTube video URL to generate a PDF with its transcription and summary.")

# Initialize session state for caching results
if 'transcription' not in st.session_state:
    st.session_state.transcription = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None
if 'video_details' not in st.session_state:
    st.session_state.video_details = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

url = st.text_input("YouTube Video URL:")

if url:
    video_id = extract_video_id(url)
    if not video_id:
        st.error("Invalid YouTube URL. Please enter a valid YouTube video URL.")
    else:
        # Fetch video details if not already cached or if URL changed
        if (not st.session_state.video_details or 
            st.session_state.video_details.get('video_id') != video_id):
            
            with st.spinner("Fetching video details..."):
                video_details, error = get_video_details(video_id)
                
            if video_details:
                st.session_state.video_details = video_details
            else:
                st.error(f"Error: {error}")
        
        # Display video information
        if st.session_state.video_details:
            video_details = st.session_state.video_details
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(video_details["thumbnail"], width=200)
            with col2:
                st.subheader(video_details["title"])
                st.write(f"**Channel:** {video_details['channel']}")
                
                # Format the date safely
                try:
                    published_date = datetime.datetime.strptime(
                        video_details['published_at'], 
                        '%Y-%m-%dT%H:%M:%SZ'
                    ).strftime('%B %d, %Y')
                except:
                    published_date = video_details['published_at']
                    
                st.write(f"**Published:** {published_date}")
                st.write(f"**Views:** {video_details['views']:,}")
                st.write(f"**Likes:** {video_details['likes']:,}")
                st.write(f"**Comments:** {video_details['comments']:,}")
            
            # Process button
            if st.button("Generate Transcription PDF"):
                st.session_state.processing = True
                
                # Step 1: Download audio
                with st.spinner("Downloading audio (this may take a few minutes)..."):
                    audio_path, error = download_youtube_audio(url)
                
                if not audio_path:
                    st.error(f"Error: {error}")
                    st.session_state.processing = False
                else:
                    # Step 2: Transcribe audio
                    with st.spinner("Transcribing audio..."):
                        transcription, error = transcribe_audio(audio_path)
                    
                    if not transcription:
                        st.error(f"Error: {error}")
                        st.session_state.processing = False
                    else:
                        st.session_state.transcription = transcription
                        
                        # Step 3: Generate summary
                        with st.spinner("Generating summary..."):
                            summary, error = generate_summary(transcription)
                        
                        if not summary and error:
                            st.warning(f"Summary generation issue: {error}")
                            # Continue without summary
                            summary = "Summary generation failed or skipped."
                        
                        st.session_state.summary = summary
                        
                        # Step 4: Create PDF
                        with st.spinner("Creating PDF..."):
                            pdf_path, error = create_pdf(
                                video_details, 
                                transcription, 
                                summary
                            )
                        
                        if not pdf_path:
                            st.error(f"Error creating PDF: {error}")
                            st.session_state.processing = False
                        else:
                            st.session_state.pdf_path = pdf_path
                            st.session_state.processing = False
            
            # Display results if processing is complete
            if st.session_state.transcription and st.session_state.pdf_path:
                with open(st.session_state.pdf_path, "rb") as f:
                    st.download_button(
                        label="Download PDF",
                        data=f,
                        file_name=f"{video_details['title'][:30]}_transcription.pdf",
                        mime="application/pdf"
                    )
                
                st.success("PDF generated successfully!")
                
                # Show transcription preview
                with st.expander("Transcription Preview"):
                    st.text_area(
                        "Transcription", 
                        st.session_state.transcription, 
                        height=300
                    )
                
                # Show summary if available
                if st.session_state.summary:
                    with st.expander("Summary"):
                        st.write(st.session_state.summary)


