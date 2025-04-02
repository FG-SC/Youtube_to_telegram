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
import yt_dlp  # More reliable alternative to pytube
from urllib.error import HTTPError

# Load environment variables
load_dotenv()

# Initialize the YouTube API client
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Whisper model (load it once and cache)
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")  # You can use "small", "medium", or "large" for better quality

model = load_whisper_model()

# Function to clean text for PDF
def clean_text(text):
    # Replace common problematic Unicode characters
    replacements = {
        '’': "'",
        '“': '"',
        '”': '"',
        '–': '-',
        '—': '-',
        '…': '...'
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text

# Function to extract video ID from URL
def extract_video_id(url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

# Function to get video details
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

# Primary download function using yt-dlp (more reliable)
def download_youtube_audio(url):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'force_generic_extractor': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_path = ydl.prepare_filename(info).replace('.webm', '.mp3').replace('.m4a', '.mp3')
            return audio_path
            
    except Exception as e:
        st.warning(f"yt-dlp failed, trying pytube as fallback: {str(e)}")
        return download_with_pytube(url)

# Fallback download function using pytube
def download_with_pytube(url):
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
            return tmp_file.name
            
    except HTTPError as e:
        if e.code == 403:
            st.error("YouTube is rate limiting us. Please try again later.")
        else:
            st.error(f"HTTP Error {e.code}: {e.reason}")
    except Exception as e:
        st.error(f"Failed to download audio: {str(e)}")
    return None

# Function to transcribe audio using Whisper
def transcribe_audio(audio_path):
    try:
        result = model.transcribe(audio_path)
        return clean_text(result["text"])
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None

# Function to generate summary using OpenAI
def generate_summary(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes video content."},
                {"role": "user", "content": f"Create a concise summary (about 500 words) of the following video transcription if it has more than a thousand words, if not, make the summary about 100 words:\n\n{text}"}
            ]
        )
        return clean_text(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

# Function to create PDF with Unicode support
def create_pdf(video_details, transcription, summary):
    pdf = FPDF()
    pdf.add_page()
    
    # Try to add Unicode font
    try:
        # Try to use DejaVuSans if available
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=12)
    except:
        try:
            # Fallback to Arial Unicode if available
            pdf.add_font("ArialUnicode", "", "arialuni.ttf", uni=True)
            pdf.set_font("ArialUnicode", size=12)
        except:
            # Final fallback to Helvetica (will have issues with special chars)
            pdf.set_font("helvetica", size=12)
            st.warning("Unicode font not found. Some special characters may not display correctly.")
    
    # Title
    pdf.set_font(size=16, style="B")
    pdf.cell(200, 10, txt=clean_text(video_details["title"]), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
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
    pdf.multi_cell(0, 10, txt=details)
    pdf.ln(10)
    
    # Summary
    pdf.set_font(size=12, style="B")
    pdf.cell(200, 10, txt="Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(size=12)
    pdf.multi_cell(0, 10, txt=summary)
    pdf.ln(10)
    
    # Transcription
    pdf.set_font(size=12, style="B")
    pdf.cell(200, 10, txt="Full Transcription", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(size=12)
    
    # Split transcription into chunks to avoid memory issues
    chunk_size = 1000
    transcription_chunks = [transcription[i:i+chunk_size] for i in range(0, len(transcription), chunk_size)]
    
    for chunk in transcription_chunks:
        pdf.multi_cell(0, 10, txt=chunk)
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
                        os.unlink(audio_path)  # Delete temporary audio file
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
                            os.unlink(pdf_path)  # Delete temporary PDF file
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
                
                If the video is age-restricted, you may need to:
                1. Sign in to YouTube in your browser
                2. Watch the video once
                3. Try again with this tool
                """)

