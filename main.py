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
import requests
import logging
import subprocess
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
from textblob import TextBlob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
st.set_page_config(page_title="YouTube Analyzer", layout="wide")

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
        return whisper.load_model("tiny")  # You can use "small", "medium", or "large" for better quality
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
        # Create temp directory for downloads
        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, "audio.wav")
        
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
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            original_path = ydl.prepare_filename(info).replace('.webm', '.wav').replace('.m4a', '.wav')
            
            # Ensure the file exists
            if not os.path.exists(original_path):
                raise FileNotFoundError(f"Downloaded file not found at {original_path}")
            
            # Convert to proper WAV format if needed
            converted_path = convert_to_wav(original_path, temp_audio_path)
            
            if not os.path.exists(converted_path):
                raise FileNotFoundError(f"Converted file not found at {converted_path}")
            
            return converted_path
            
    except Exception as e:
        logger.error(f"Download failed: {e}")
        try:
            # Clean up temp files if they exist
            if 'original_path' in locals() and os.path.exists(original_path):
                os.unlink(original_path)
            if 'converted_path' in locals() and os.path.exists(converted_path):
                os.unlink(converted_path)
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except:
            pass
        
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
            model="gpt-3.5-turbo",
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

# Channel Analysis Functions
def get_channel_id(channel_name):
    try:
        request = youtube.search().list(
            part="snippet",
            q=channel_name,
            type="channel",
            maxResults=1
        )
        response = request.execute()

        if response["items"]:
            return response["items"][0]["snippet"]["channelId"]
        return None
    except Exception as e:
        st.error(f"Error retrieving channel ID: {e}")
        return None

def get_uploads_playlist_id(channel_id):
    try:
        request = youtube.channels().list(
            part="contentDetails",
            id=channel_id
        )
        response = request.execute()

        if response["items"]:
            return response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        return None
    except Exception as e:
        st.error(f"Error retrieving uploads playlist ID: {e}")
        return None

def get_all_video_ids(playlist_id, max_results=10):
    video_ids = []
    next_page_token = None

    while True:
        try:
            request = youtube.playlistItems().list(
                part="contentDetails",
                playlistId=playlist_id,
                maxResults=min(50, max_results - len(video_ids)),
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response["items"]:
                video_ids.append(item["contentDetails"]["videoId"])
                if len(video_ids) >= max_results:
                    break

            # Check if there are more pages
            next_page_token = response.get("nextPageToken")
            if not next_page_token or len(video_ids) >= max_results:
                break
        except Exception as e:
            st.error(f"Error retrieving video IDs: {e}")
            break

    return video_ids

def get_video_details_batch(video_ids):
    video_data = []
    for i in range(0, len(video_ids), 50):  # Process 50 videos at a time (API limit)
        try:
            batch_ids = video_ids[i:i + 50]
            request = youtube.videos().list(
                part="snippet,statistics",
                id=",".join(batch_ids)
            )
            response = request.execute()

            for item in response["items"]:
                snippet = item["snippet"]
                statistics = item["statistics"]
                video_data.append({
                    "video_id": item["id"],
                    "title": snippet["title"],
                    "published_at": snippet["publishedAt"],
                    "views": int(statistics.get("viewCount", 0)),
                    "likes": int(statistics.get("likeCount", 0)),
                    "comments": int(statistics.get("commentCount", 0)),
                    "thumbnail": snippet["thumbnails"]["high"]["url"]
                })
        except Exception as e:
            st.error(f"Error retrieving video details: {e}")

    return video_data

def get_channel_stats(channel_id):
    try:
        request = youtube.channels().list(
            part="statistics",
            id=channel_id
        )
        response = request.execute()

        if response["items"]:
            stats = response["items"][0]["statistics"]
            return {
                "subscribers": int(stats.get("subscriberCount", 0)),
                "views": int(stats.get("viewCount", 0)),
                "videos": int(stats.get("videoCount", 0)),
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        return None
    except Exception as e:
        st.error(f"Error retrieving channel stats: {e}")
        return None

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        output = ' '.join([x['text'] for x in transcript])
        return clean_text(output)
    except Exception as e:
        st.warning(f"Could not get transcript for video {video_id}: {e}")
        return None

def summarize_transcription(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a journalist."},
                {"role": "assistant", "content": "Write a 100-word summary of this video talking about the main topics discussed."},
                {"role": "user", "content": text}
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

def generate_tags(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a journalist."},
                {"role": "assistant", "content": "Output a list of the main tags for this blog post in a Python list such as ['item1', 'item2', 'item3']. Don't be repetitive."},
                {"role": "user", "content": text}
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Error generating tags: {e}")
        return None

def create_channel_pdf(channel_name, channel_stats, videos_data):
    pdf = FPDF()
    pdf.add_page()
    
    # Try to add Unicode font
    try:
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=12)
    except:
        pdf.set_font("helvetica", size=12)
    
    # Channel header
    pdf.set_font(size=16, style="B")
    pdf.cell(200, 10, txt=f"Channel Report: {channel_name}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(10)
    
    # Channel stats
    pdf.set_font(size=12, style="B")
    pdf.cell(200, 10, txt="Channel Statistics", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(size=12)
    
    stats_text = f"""Subscribers: {channel_stats['subscribers']:,}
Total Views: {channel_stats['views']:,}
Total Videos: {channel_stats['videos']:,}
Report Generated: {channel_stats['timestamp']}
"""
    pdf.multi_cell(0, 10, txt=stats_text)
    pdf.ln(10)
    
    # Videos section
    pdf.set_font(size=14, style="B")
    pdf.cell(200, 10, txt="Recent Videos Analysis", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)
    
    for video in videos_data:
        pdf.set_font(size=12, style="B")
        pdf.multi_cell(0, 10, txt=clean_text(video['title']))
        pdf.set_font(size=10)
        
        details = f"""Published: {datetime.datetime.strptime(video['published_at'], '%Y-%m-%dT%H:%M:%SZ').strftime('%B %d, %Y')}
Views: {video['views']:,} | Likes: {video['likes']:,} | Comments: {video['comments']:,}
"""
        pdf.multi_cell(0, 8, txt=details)
        
        if video.get('summary'):
            pdf.set_font(size=10, style="B")
            pdf.cell(0, 8, txt="Summary:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font(size=10)
            pdf.multi_cell(0, 8, txt=clean_text(video['summary']))
        
        if video.get('tags'):
            pdf.set_font(size=10, style="B")
            pdf.cell(0, 8, txt="Tags:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font(size=10)
            pdf.multi_cell(0, 8, txt=clean_text(video['tags']))
        
        pdf.ln(5)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
    
    # Save to temporary file
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_pdf.name)
    return temp_pdf.name

# Streamlit app

st.title("YouTube Video Analyzer")
st.write("Analyze individual videos or entire channels")

# Initialize session state variables
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None
if 'transcription' not in st.session_state:
    st.session_state.transcription = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'video_details' not in st.session_state:
    st.session_state.video_details = None
if 'channel_pdf_path' not in st.session_state:
    st.session_state.channel_pdf_path = None
if 'videos_data' not in st.session_state:
    st.session_state.videos_data = None
if 'channel_stats' not in st.session_state:
    st.session_state.channel_stats = None

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Single Video Analysis", "Channel Analysis"])

with tab1:
    st.header("Single Video Analysis")
    url = st.text_input("Enter YouTube Video URL:", key="video_url")
    
    if url:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Invalid YouTube URL. Please enter a valid YouTube video URL.")
        
        # Only fetch details if we don't have them or URL changed
        if not st.session_state.video_details or st.session_state.video_details.get('id') != video_id:
            with st.spinner("Fetching video details..."):
                st.session_state.video_details = get_video_details(video_id)
                if st.session_state.video_details:
                    st.session_state.video_details['id'] = video_id  # Store ID for comparison
        
        if st.session_state.video_details:
            video_details = st.session_state.video_details
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
                        st.session_state.transcription = transcribe_audio(audio_path)
                        try:
                            os.unlink(audio_path)  # Delete temporary audio file
                        except:
                            pass
                    
                    if st.session_state.transcription:
                        with st.spinner("Generating summary..."):
                            st.session_state.summary = generate_summary(st.session_state.transcription)
                        
                        if st.session_state.summary:
                            with st.spinner("Creating PDF..."):
                                st.session_state.pdf_path = create_pdf(video_details, st.session_state.transcription, st.session_state.summary)
                            
                            # Display the results and buttons
                            display_results = True
        
        # Show results if we have them
        if st.session_state.pdf_path and st.session_state.transcription and st.session_state.summary:
            # Display download button
            with open(st.session_state.pdf_path, "rb") as f:
                st.download_button(
                    label="Download PDF",
                    data=f,
                    file_name=f"{video_details['title']}_transcription.pdf",
                    mime="application/pdf"
                )
            
            st.success("PDF generated successfully!")
            st.text_area("Transcription Preview", st.session_state.transcription, height=300)

with tab2:
    st.header("Channel Analysis")
    channel_input = st.text_input("Enter YouTube Channel Name or URL:", key="channel_name")
    num_videos = st.slider("Number of recent videos to analyze:", 1, 20, 5)
    
    if st.button("Analyze Channel"):
        if not channel_input:
            st.error("Please enter a channel name or URL")
        else:
            with st.spinner(f"Analyzing channel..."):
                try:
                    # Handle both URL (@handle) and channel name inputs
                    if "@" in channel_input:
                        # Extract handle from URL (e.g., https://www.youtube.com/@brazilcryptoreport)
                        handle = channel_input.split('@')[-1].split('/')[0]
                        search_query = f"@{handle}"
                    else:
                        search_query = channel_input
                    
                    # Step 1: Get channel ID
                    channel_id = get_channel_id(search_query)
                    if not channel_id:
                        st.error(f"Channel '{channel_input}' not found. Please check the name/URL and try again.")
                        st.stop()
                    
                    # Get channel stats
                    st.session_state.channel_stats = get_channel_stats(channel_id)
                    if not st.session_state.channel_stats:
                        st.error("Failed to get channel statistics")
                        st.stop()
                    
                    # Get uploads playlist
                    playlist_id = get_uploads_playlist_id(channel_id)
                    if not playlist_id:
                        st.error("Could not find uploads playlist for this channel")
                        st.stop()
                    
                    # Get recent videos
                    video_ids = get_all_video_ids(playlist_id, num_videos)
                    if not video_ids:
                        st.error("No videos found for this channel")
                        st.stop()
                    
                    # Get video details
                    st.session_state.videos_data = get_video_details_batch(video_ids)
                    if not st.session_state.videos_data:
                        st.error("Could not get video details")
                        st.stop()
                    
                    # Process each video (transcript, summary, tags)
                    progress_bar = st.progress(0)
                    for i, video in enumerate(st.session_state.videos_data):
                        progress_bar.progress((i + 1) / len(st.session_state.videos_data))
                        
                        # Get transcript
                        transcript = get_transcript(video['video_id'])
                        if transcript:
                            video['transcript'] = transcript
                            
                            # Generate summary
                            summary = summarize_transcription(transcript)
                            video['summary'] = summary if summary else "No summary available"
                            
                            # Generate tags
                            tags = generate_tags(transcript)
                            video['tags'] = tags if tags else "No tags generated"
                    
                    # Create PDF report
                    st.session_state.channel_pdf_path = create_channel_pdf(
                        channel_input,
                        st.session_state.channel_stats,
                        st.session_state.videos_data
                    )
                    
                    # Display results
                    st.success("Channel analysis complete!")
                    
                    # Show channel stats
                    st.subheader("Channel Statistics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Subscribers", f"{st.session_state.channel_stats['subscribers']:,}")
                    col2.metric("Total Views", f"{st.session_state.channel_stats['views']:,}")
                    col3.metric("Total Videos", f"{st.session_state.channel_stats['videos']:,}")
                    
                    # Show video list
                    st.subheader(f"Recent Videos (Last {num_videos})")
                    for video in st.session_state.videos_data:
                        with st.expander(f"{video['title']} - {datetime.datetime.strptime(video['published_at'], '%Y-%m-%dT%H:%M:%SZ').strftime('%b %d, %Y')}"):
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.image(video['thumbnail'], width=150)
                            with col2:
                                st.write(f"**Views:** {video['views']:,} | **Likes:** {video['likes']:,} | **Comments:** {video['comments']:,}")
                                if 'summary' in video:
                                    st.write("**Summary:**")
                                    st.write(video['summary'])
                                if 'tags' in video:
                                    st.write("**Tags:**")
                                    st.write(video['tags'])
                    
                    # Download button
                    with open(st.session_state.channel_pdf_path, "rb") as f:
                        st.download_button(
                            label="Download Full Channel Report",
                            data=f,
                            file_name=f"{channel_input}_channel_report.pdf",
                            mime="application/pdf"
                        )
                
                except Exception as e:
                    st.error(f"An error occurred during channel analysis: {str(e)}")
                    logger.error(f"Channel analysis error: {str(e)}")
