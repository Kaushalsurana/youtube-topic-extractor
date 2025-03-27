import streamlit as st
import openai
import os
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import time

# --- Helper Functions ---

def format_time(seconds):
    """Converts seconds to HH:MM:SS or MM:SS format."""
    if seconds is None: return "00:00"
    try: seconds = float(seconds)
    except (ValueError, TypeError): return "??:??" # Handle unexpected non-numeric input
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02}:{minutes:02}:{secs:02}"
    else:
        return f"{minutes:02}:{secs:02}"

def get_video_id(url):
    """Extracts YouTube video ID from various URL formats."""
    if not isinstance(url, str): # Basic type check
        return None
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^?]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^?]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([^?]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # Check if the input itself might be just the ID
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
        return url
    return None

def get_transcript(video_id):
    """Fetches the transcript, displaying status in Streamlit."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Define preferred languages (can be customized)
        preferred_languages = [
            'en', 'en-US', 'en-GB', 'hi', 'es', 'fr', 'de', 'pt', 'it', 'ru',
            'ja', 'ko', 'zh-Hans', 'zh-CN', 'ar', 'id', 'tr', 'vi'
        ]
        transcript = None
        found_lang = None
        transcript_type = ""

        # Try manual first
        for lang in preferred_languages:
            try:
                transcript = transcript_list.find_manually_created_transcript([lang])
                found_lang = lang
                transcript_type = "manually created"
                st.info(f"Found {transcript_type} transcript ({found_lang}).")
                break
            except NoTranscriptFound: continue

        # Try generated if no manual found
        if not transcript:
            for lang in preferred_languages:
                try:
                    transcript = transcript_list.find_generated_transcript([lang])
                    found_lang = lang
                    transcript_type = "auto-generated"
                    st.info(f"Found {transcript_type} transcript ({found_lang}).")
                    break
                except NoTranscriptFound: continue

        # Handle case where no preferred transcript is found
        if not transcript:
            try:
                available_langs = [f"{t.language} ({'manual' if t.is_translatable else 'generated' if t.is_generated else 'unknown'})" for t in transcript_list]
                if available_langs:
                    st.warning(f"No transcript found in preferred languages.")
                    st.caption(f"Available transcripts: {', '.join(available_langs)}")
                else:
                    st.error(f"No transcripts found at all for video ID: {video_id}")
                return None
            except Exception as e:
                st.error(f"Error listing available transcripts: {e}")
                return None

        # Fetch data
        st.info(f"Fetching transcript content for {found_lang} ({transcript_type})...")
        transcript_data = transcript.fetch()
        return transcript_data

    except TranscriptsDisabled:
        st.error(f"Transcripts are disabled for video ID: {video_id}")
        return None
    except Exception as e:
        # Catch more general errors during the process
        st.error(f"An unexpected error occurred fetching transcript: {e}")
        return None


def format_transcript_for_llm(transcript_data):
    """Formats transcript data into a string suitable for LLM processing."""
    formatted = []
    if not transcript_data: return ""
    for segment in transcript_data:
        # Use dot notation, with fallback for safety
        try:
            time_val = segment.start
            text_val = segment.text
        except AttributeError:
             try:
                 time_val = segment['start']
                 text_val = segment['text']
             except (KeyError, TypeError):
                 st.warning(f"Skipping segment due to unexpected format: {segment}")
                 continue # Skip malformed segment

        time_str = format_time(time_val)
        # Ensure text is treated as string and handle potential None
        text = str(text_val or '').replace('\n', ' ').strip()
        formatted.append(f"[{time_str}] {text}")
    return "\n".join(formatted)


def get_topics_with_llm(client, transcript_string):
    """Sends transcript to the LLM (gpt-4o-mini), shows status in Streamlit."""
    if not client:
        st.error("OpenAI client not initialized.")
        return None
    if not transcript_string:
        st.warning("Transcript string is empty. Cannot send to LLM.")
        return None

    # Define the model to use
    model_to_use = "gpt-4o-mini"

    system_prompt = """You are an expert video analyst. Your task is to identify the main topics discussed in the provided YouTube video transcript and the timestamp when each topic begins.
Use the timestamps provided in the transcript (format [HH:MM:SS] or [MM:SS]) as reference points.
List each main topic with a concise title and its corresponding start timestamp.
Format the output clearly, with each topic on a new line like this:
Topic Title: HH:MM:SS
Ensure timestamps accurately reflect the start of the topic based on the provided transcript timings. Respond only with the list of topics and timestamps.
"""

    user_prompt = f"""Here is the video transcript with timestamps:

{transcript_string}

---
Please analyze this transcript and provide the main topics covered and their start timestamps in the format:
Topic Title: HH:MM:SS
"""

    try:
        st.write(f"Sending transcript to {model_to_use} for topic identification...")

        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5, # Can adjust if needed
            max_tokens=800   # Max tokens for the response
        )

        # Check for valid response structure
        if response.choices and response.choices[0].message and response.choices[0].message.content:
             content = response.choices[0].message.content
             # Check finish reason for potential issues
             finish_reason = response.choices[0].finish_reason
             if finish_reason == 'content_filter':
                 st.warning("OpenAI response was flagged by content filters.")
             elif finish_reason == 'length':
                 st.warning(f"OpenAI response may be truncated due to max_tokens limit. Consider increasing max_tokens if topics seem cut off.")
             return content
        else:
            st.error("Received an empty or incomplete response from OpenAI.")
            st.json(response) # Show the raw response object for debugging
            return None

    except openai.AuthenticationError:
        st.error("OpenAI Authentication Error: Invalid API Key.")
        return None
    except openai.RateLimitError:
        st.error("OpenAI Rate Limit Exceeded. Please wait or check your OpenAI plan.")
        return None
    except openai.APITimeoutError:
        st.error("OpenAI API request timed out. Please try again later.")
        return None
    except openai.NotFoundError:
        st.error(f"Model '{model_to_use}' not found. Check the model name or your API key permissions.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during the OpenAI API call: {e}")
        return None

def parse_llm_response(response_text):
    """Parses the LLM response to extract topics and timestamps."""
    if not response_text:
        return []
    topics = []
    # Regex to find lines like "Topic Title: HH:MM:SS" or "Topic Title: MM:SS"
    # Allows variations in spacing and potential markdown emphasis (* or **)
    pattern = re.compile(r"^\s*(?:\*?\*?)(.*?)(?:\*?\*?)\s*:\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*$", re.MULTILINE)
    matches = pattern.findall(response_text)

    if not matches:
        st.warning("Could not parse LLM response using primary regex. Attempting simple line split.")
        lines = response_text.strip().split('\n')
        fallback_found = False
        for line in lines:
            line = line.strip()
            if not line: continue # Skip empty lines
            if ':' in line:
                parts = line.split(':', 1)
                topic = parts[0].strip().strip('*').strip() # Clean up potential markdown
                timestamp = parts[1].strip()
                # Basic validation of timestamp format
                if re.match(r"^\d{1,2}:\d{2}(?::\d{2})?$", timestamp):
                    topics.append({"topic": topic, "timestamp": timestamp})
                    fallback_found = True
                else:
                    st.caption(f"Skipping line (invalid timestamp format): {line}")
            else:
                 st.caption(f"Skipping line (no colon found): {line}")
        if not fallback_found:
             st.error("Failed to parse topics using fallback method either. Check raw LLM response.")
    else:
        for match in matches:
            # Clean up topic title from potential extra spaces or markdown
            topic_title = match[0].strip().strip('*').strip()
            topics.append({"topic": topic_title, "timestamp": match[1].strip()})

    return topics


# --- Streamlit App ---

st.set_page_config(page_title="YouTube Topic Extractor", layout="wide")
st.title("🎥 YouTube Video Topic Extractor")
st.caption("Enter a YouTube URL to get timestamped topics (using GPT-4o Mini).")

# --- API Key Handling (Prioritize Secrets for Deployment) ---
st.sidebar.header("Configuration")
api_key = None
client = None # Initialize client as None

# 1. Try Streamlit secrets (Ideal for deployment)
try:
    # Check if secrets is available and key exists
    if hasattr(st, 'secrets') and 'openai_api_key' in st.secrets:
        api_key = st.secrets["openai_api_key"]
        st.sidebar.info("Using API Key from Streamlit Secrets.")
except Exception as e:
    # Handle potential errors accessing secrets (e.g., not available locally)
    # st.sidebar.caption(f"Secrets not accessible: {e}") # Optional debug info
    pass

# 2. Fallback to manual input if secrets are not found/configured
if not api_key:
    st.sidebar.warning("OpenAI API Key not found in Secrets.")
    api_key = st.sidebar.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        help="Configure 'openai_api_key' in Streamlit Cloud secrets for deployed apps."
    )

# --- YouTube URL Input ---
youtube_url = st.text_input("Enter the YouTube video URL:", placeholder="e.g., https://www.youtube.com/watch?v=...")

# --- Analyze Button ---
# Disable button if URL is missing OR if manual API key input is required but not provided
button_disabled = not youtube_url or not api_key
analyze_button = st.button("Analyze Video", disabled=button_disabled, type="primary")

# --- User Guidance Messages ---
if not youtube_url and not api_key:
    st.info("Please provide your OpenAI API Key (either via Secrets or input) and enter a YouTube URL.")
elif not api_key:
    st.info("Please provide your OpenAI API Key (either via Secrets or input).")
elif not youtube_url:
    st.info("Please enter a YouTube URL.")


# --- Processing Logic (Only runs if button clicked AND api_key is available) ---
if analyze_button and api_key:
    # Initialize OpenAI Client here, using the obtained api_key
    try:
        client = openai.OpenAI(api_key=api_key)
        # Quick check to validate the key before proceeding
        client.models.list()
        st.sidebar.success("OpenAI client ready.")
    except openai.AuthenticationError:
        st.error("Authentication Error: The provided OpenAI API Key is invalid. Please check the key in Secrets or input.")
        client = None # Ensure client is None on error
        st.stop() # Stop execution if key is invalid
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        client = None # Ensure client is None on error
        st.stop() # Stop execution on other init errors

    # Proceed only if client initialization was successful
    if client:
        # Define variables to hold results outside the status block
        analysis_successful = False
        parsed_topics_result = None
        llm_response_result = None
        error_message = None

        video_id = get_video_id(youtube_url)

        if not video_id:
            st.error("Invalid YouTube URL or could not extract Video ID.")
        else:
            st.subheader(f"Processing Video ID: `{video_id}`")

            with st.status("Analyzing video...", expanded=True) as status:
                # 1. Get Transcript
                st.write("Fetching transcript...")
                transcript_data = get_transcript(video_id)

                if transcript_data:
                    # 2. Format Transcript
                    st.write("Formatting transcript...")
                    formatted_transcript = format_transcript_for_llm(transcript_data)

                    if formatted_transcript:
                        # 3. Get Topics from LLM
                        # Pass the initialized client
                        llm_response = get_topics_with_llm(client, formatted_transcript)
                        llm_response_result = llm_response # Store for potential later display

                        if llm_response:
                            # 4. Parse LLM Response
                            st.write("Parsing topics from response...")
                            parsed_topics = parse_llm_response(llm_response)

                            if parsed_topics:
                                # Success Case
                                status.update(label="Analysis Complete!", state="complete", expanded=False)
                                parsed_topics_result = parsed_topics # Store results
                                analysis_successful = True
                            else:
                                # Parsing Failed Case
                                status.update(label="Parsing Failed", state="error", expanded=True)
                                error_message = "Could not parse topics from the LLM response."
                                # llm_response_result is already stored
                        else:
                            # LLM Call Failed Case
                            status.update(label="LLM Analysis Failed", state="error", expanded=False)
                            error_message = "Failed to get response from LLM." # Specific error likely shown by get_topics_with_llm
                    else:
                        # Formatting Failed Case
                        status.update(label="Formatting Failed", state="error", expanded=False)
                        error_message = "Formatted transcript is empty, possibly due to issues during formatting."
                else:
                    # Transcript Fetch Failed Case
                    status.update(label="Transcript Fetch Failed", state="error", expanded=False)
                    error_message = "Could not retrieve transcript." # Specific error likely shown by get_transcript

        # --- Display Results (Outside and After Status Block) ---
        if analysis_successful and parsed_topics_result:
            st.subheader("✅ Identified Topics and Timestamps")
            # Use columns for potentially better layout if many topics
            # col1, col2 = st.columns(2) # Example: 2 columns
            for i, item in enumerate(parsed_topics_result):
                # with col1 if i % 2 == 0 else col2: # Example: distribute in columns
                st.markdown(f"- **{item['topic']}**: `{item['timestamp']}`")

            # Expander for raw response, only shown on success
            if llm_response_result:
                 with st.expander("View Raw LLM Response"):
                    st.text_area("Raw Response", llm_response_result, height=200, disabled=True) # Use text_area for scrollable content

        elif error_message:
            # Display final error summary if analysis didn't complete successfully
            st.error(f"Analysis failed: {error_message}")
            # Optionally show raw response even on parsing failure, if available
            if llm_response_result and "Could not parse topics" in error_message:
                 st.subheader("Raw LLM Response (Parsing Failed):")
                 st.text_area("Raw Response", llm_response_result, height=200, disabled=True)

# --- End of App ---