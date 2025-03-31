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
import asyncio
import nest_asyncio

# Patch the event loop
try:
    nest_asyncio.apply()
except:
    pass
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize services
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
def verify_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              text=True)
        return True
    except:
        return False

if not verify_ffmpeg():
    st.warning("FFmpeg not properly installed. Some features may not work.")
    
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

# Audio processing functions
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
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e}")
        return None

# Modified download function with format conversion
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
        '¼': '1/4', '½': '1/2', '¾': '3/4', '×': 'x', '÷': '/', '‰': '0/00',
        '€': 'EUR', '£': 'GBP', '¥': 'JPY', '¢': 'c', '¤': '$', '¦': '|',
        '§': 'S', '¨': '"', 'ª': 'a', '¬': '-', '¯': '-', '´': "'", '¸': ',',
        'º': 'o', '¿': '?', 'À': 'A', 'Á': 'A', 'Â': 'A', 'Ã': 'A', 'Ä': 'Ae',
        'Å': 'A', 'Æ': 'AE', 'Ç': 'C', 'È': 'E', 'É': 'E', 'Ê': 'E', 'Ë': 'E',
        'Ì': 'I', 'Í': 'I', 'Î': 'I', 'Ï': 'I', 'Ð': 'D', 'Ñ': 'N', 'Ò': 'O',
        'Ó': 'O', 'Ô': 'O', 'Õ': 'O', 'Ö': 'Oe', 'Ø': 'O', 'Ù': 'U', 'Ú': 'U',
        'Û': 'U', 'Ü': 'Ue', 'Ý': 'Y', 'Þ': 'TH', 'ß': 'ss', 'à': 'a', 'á': 'a',
        'â': 'a', 'ã': 'a', 'ä': 'ae', 'å': 'a', 'æ': 'ae', 'ç': 'c', 'è': 'e',
        'é': 'e', 'ê': 'e', 'ë': 'e', 'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
        'ð': 'd', 'ñ': 'n', 'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'oe',
        'ø': 'o', 'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'ue', 'ý': 'y', 'þ': 'th',
        'ÿ': 'y'
    }
    
    # First pass - standard replacements
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    
    # Second pass - handle emojis and other special chars
    emoji_replacements = {
        '🌎': '[Globe]', '🌍': '[Globe]', '🌏': '[Globe]',
        '🔥': '[Hot]', '❤️': '[Heart]', '✅': '[Check]',
        '⚠️': '[Warning]', '⚡': '[Lightning]', '✨': '[Sparkle]',
        '🎯': '[Target]', '📈': '[Chart Up]', '📉': '[Chart Down]',
        '📊': '[Chart]', '📌': '[Pin]', '📍': '[Location]',
        '📝': '[Notes]', '🔍': '[Search]', '🔎': '[Magnifying Glass]',
        '🔑': '[Key]', '🔔': '[Bell]', '🚀': '[Rocket]',
        '🛑': '[Stop]', '🤔': '[Thinking]', '💰': '[Money]',
        '🔄': '[Refresh]', '🖥️': '[Computer]', '🖱️': '[Mouse]',
        '🗂️': '[Folder]', '🗒️': '[Notepad]', '🗓️': '[Calendar]',
        '😊': '[Smile]', '😃': '[Happy]', '😎': '[Cool]',
        '🙏': '[Thanks]', '👍': '[Thumbs Up]', '👎': '[Thumbs Down]',
        '👏': '[Clap]', '💡': '[Idea]', '💻': '[Laptop]',
        '💾': '[Save]', '📁': '[Folder]', '📂': '[Open Folder]',
        '📅': '[Calendar]', '📆': '[Tear-off Calendar]',
        '📌': '[Pushpin]', '📍': '[Round Pushpin]',
        '📎': '[Paperclip]', '📏': '[Ruler]', '📐': '[Triangular Ruler]',
        '📒': '[Ledger]', '📓': '[Notebook]', '📔': '[Notebook with Decorative Cover]',
        '📕': '[Closed Book]', '📗': '[Green Book]', '📘': '[Blue Book]',
        '📙': '[Orange Book]', '📚': '[Books]', '📛': '[Name Badge]',
        '📜': '[Scroll]', '📝': '[Memo]', '📞': '[Telephone Receiver]',
        '📟': '[Pager]', '📠': '[Fax Machine]', '📡': '[Satellite Antenna]',
        '📢': '[Loudspeaker]', '📣': '[Megaphone]', '📤': '[Outbox Tray]',
        '📥': '[Inbox Tray]', '📦': '[Package]', '📧': '[E-mail]',
        '📨': '[Incoming Envelope]', '📩': '[Envelope with Arrow]',
        '📪': '[Closed Mailbox with Lowered Flag]', '📫': '[Closed Mailbox with Raised Flag]',
        '📬': '[Open Mailbox with Raised Flag]', '📭': '[Open Mailbox with Lowered Flag]',
        '📮': '[Postbox]', '📯': '[Postal Horn]', '📰': '[Newspaper]',
        '📱': '[Mobile Phone]', '📲': '[Mobile Phone with Rightwards Arrow at Left]',
        '📳': '[Vibration Mode]', '📴': '[Mobile Phone Off]', '📶': '[Antenna with Bars]',
        '📷': '[Camera]', '📸': '[Camera with Flash]', '📹': '[Video Camera]',
        '📺': '[Television]', '📻': '[Radio]', '📼': '[Videocassette]',
        '📽️': '[Film Projector]', '📿': '[Prayer Beads]', '🔀': '[Twisted Rightwards Arrows]',
        '🔁': '[Clockwise Rightwards and Leftwards Open Circle Arrows]',
        '🔂': '[Clockwise Rightwards and Leftwards Open Circle Arrows with Circled One Overlay]',
        '🔃': '[Clockwise Downwards and Upwards Open Circle Arrows]',
        '🔄': '[Anticlockwise Downwards and Upwards Open Circle Arrows]',
        '🔅': '[Low Brightness Symbol]', '🔆': '[High Brightness Symbol]',
        '🔇': '[Speaker with Cancellation Stroke]', '🔈': '[Speaker]',
        '🔉': '[Speaker with One Sound Wave]', '🔊': '[Speaker with Three Sound Waves]',
        '🔋': '[Battery]', '🔌': '[Electric Plug]', '🔍': '[Left-Pointing Magnifying Glass]',
        '🔎': '[Right-Pointing Magnifying Glass]', '🔏': '[Lock with Ink Pen]',
        '🔐': '[Closed Lock with Key]', '🔑': '[Key]', '🔒': '[Lock]',
        '🔓': '[Open Lock]', '🔔': '[Bell]', '🔕': '[Bell with Cancellation Stroke]',
        '🔖': '[Bookmark]', '🔗': '[Link Symbol]', '🔘': '[Radio Button]',
        '🔙': '[Back with Leftwards Arrow Above]', '🔚': '[End with Leftwards Arrow Above]',
        '🔛': '[On with Exclamation Mark with Left Right Arrow Above]',
        '🔜': '[Soon with Rightwards Arrow Above]', '🔝': '[Top with Upwards Arrow Above]',
        '🔞': '[No One Under Eighteen Symbol]', '🔟': '[Keycap Ten]',
        '🔠': '[Input Symbol for Latin Capital Letters]', '🔡': '[Input Symbol for Latin Small Letters]',
        '🔢': '[Input Symbol for Numbers]', '🔣': '[Input Symbol for Symbols]',
        '🔤': '[Input Symbol for Latin Letters]', '🔥': '[Fire]', '🔦': '[Electric Torch]',
        '🔧': '[Wrench]', '🔨': '[Hammer]', '🔩': '[Nut and Bolt]',
        '🔪': '[Hocho]', '🔫': '[Pistol]', '🔬': '[Microscope]',
        '🔭': '[Telescope]', '🔮': '[Crystal Ball]', '🔯': '[Six Pointed Star with Middle Dot]',
        '🔰': '[Japanese Symbol for Beginner]', '🔱': '[Trident Emblem]',
        '🔲': '[Black Square Button]', '🔳': '[White Square Button]',
        '🔴': '[Large Red Circle]', '🔵': '[Large Blue Circle]',
        '🔶': '[Large Orange Diamond]', '🔷': '[Large Blue Diamond]',
        '🔸': '[Small Orange Diamond]', '🔹': '[Small Blue Diamond]',
        '🔺': '[Up-Pointing Red Triangle]', '🔻': '[Down-Pointing Red Triangle]',
        '🔼': '[Up-Pointing Small Red Triangle]', '🔽': '[Down-Pointing Small Red Triangle]',
        '🕉️': '[Om Symbol]', '🕊️': '[Dove of Peace]', '🕋': '[Kaaba]',
        '🕌': '[Mosque]', '🕍': '[Synagogue]', '🕎': '[Menorah with Nine Branches]',
        '🕐': '[Clock Face One Oclock]', '🕑': '[Clock Face Two Oclock]',
        '🕒': '[Clock Face Three Oclock]', '🕓': '[Clock Face Four Oclock]',
        '🕔': '[Clock Face Five Oclock]', '🕕': '[Clock Face Six Oclock]',
        '🕖': '[Clock Face Seven Oclock]', '🕗': '[Clock Face Eight Oclock]',
        '🕘': '[Clock Face Nine Oclock]', '🕙': '[Clock Face Ten Oclock]',
        '🕚': '[Clock Face Eleven Oclock]', '🕛': '[Clock Face Twelve Oclock]',
        '🕜': '[Clock Face One-Thirty]', '🕝': '[Clock Face Two-Thirty]',
        '🕞': '[Clock Face Three-Thirty]', '🕟': '[Clock Face Four-Thirty]',
        '🕠': '[Clock Face Five-Thirty]', '🕡': '[Clock Face Six-Thirty]',
        '🕢': '[Clock Face Seven-Thirty]', '🕣': '[Clock Face Eight-Thirty]',
        '🕤': '[Clock Face Nine-Thirty]', '🕥': '[Clock Face Ten-Thirty]',
        '🕦': '[Clock Face Eleven-Thirty]', '🕧': '[Clock Face Twelve-Thirty]',
    }
    
    # Replace emojis with text descriptions
    for emoji, description in emoji_replacements.items():
        text = text.replace(emoji, description)
    
    # Final cleanup - remove any remaining non-ASCII characters if they cause problems
    try:
        text.encode('ascii')
    except UnicodeEncodeError:
        # If we still have non-ASCII chars, remove them
        text = text.encode('ascii', 'ignore').decode('ascii')
    
    return text.strip()

# Modified transcribe function with better error handling
def transcribe_audio(audio_path):
    if not audio_path or not os.path.exists(audio_path):
        st.error(f"Audio file not found at: {audio_path}")
        return None
    
    try:
        # Verify file is readable
        with open(audio_path, 'rb') as f:
            pass
            
        # Load audio with Whisper
        result = model.transcribe(audio_path)
        return clean_text(result["text"])
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        logger.error(f"Transcription error: {str(e)}")
        return None

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

# Function to generate summary using OpenAI
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
    
    # Try to add Unicode font - this is CRUCIAL for emoji/special char support
    try:
        # First try DejaVuSans if available
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=12)
    except:
        try:
            # Fallback to Arial Unicode if available
            pdf.add_font("ArialUnicode", "", "arialuni.ttf", uni=True)
            pdf.set_font("ArialUnicode", size=12)
        except:
            try:
                # Try to use a different Unicode font
                pdf.add_font("NotoSans", "", "NotoSans-Regular.ttf", uni=True)
                pdf.set_font("NotoSans", size=12)
            except:
                # Final fallback - will have issues with special chars
                pdf.set_font("helvetica", size=12)
                st.warning("Unicode font not found. Some special characters may not display correctly.")
    
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

# Function to send PDF to Telegram
def send_pdf_to_telegram(pdf_path, bot_token, chat_id):
    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    with open(pdf_path, "rb") as f:
        files = {"document": f}
        data = {"chat_id": chat_id}
        response = requests.post(url, files=files, data=data)
    return response.ok

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
                        
                        # Display download button
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                label="Download PDF",
                                data=f,
                                file_name=f"{video_details['title']}_transcription.pdf",
                                mime="application/pdf"
                            )
                        
                        # Add Telegram send button if credentials are configured
                        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID:
                            if st.button("Send to Telegram"):
                                with st.spinner("Sending to Telegram..."):
                                    if send_pdf_to_telegram(pdf_path, TELEGRAM_BOT_TOKEN, TELEGRAM_CHANNEL_ID):
                                        st.success("PDF sent to Telegram successfully!")
                                    else:
                                        st.error("Failed to send PDF to Telegram.")
                        else:
                            st.warning("Telegram credentials not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID in your .env file to enable Telegram sending.")
                        
                        # Clean up temporary files
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
                
                If the video is age-restricted, you may need to:
                1. Sign in to YouTube in your browser
                2. Watch the video once
                3. Try again with this tool
                """)
