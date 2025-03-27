import streamlit as st
import openai # Still needed for error types potentially
from openai import AzureOpenAI # Import the Azure client
import os
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import time

# --- Helper Functions (Keep format_time, get_video_id, get_transcript, format_transcript_for_llm, parse_llm_response exactly the same as the previous complete version) ---

def format_time(seconds):
    # ... (same code) ...
    if seconds is None: return "00:00"
    try: seconds = float(seconds)
    except (ValueError, TypeError): return "??:??"
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0: return f"{hours:02}:{minutes:02}:{secs:02}"
    else: return f"{minutes:02}:{secs:02}"

def get_video_id(url):
    # ... (same code) ...
    if not isinstance(url, str): return None
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^?]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^?]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([^?]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match: return match.group(1)
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url): return url
    return None

def get_transcript(video_id):
    # ... (same code using st.info/st.warning/st.error) ...
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        preferred_languages = [
            'en', 'en-US', 'en-GB', 'hi', 'es', 'fr', 'de', 'pt', 'it', 'ru',
            'ja', 'ko', 'zh-Hans', 'zh-CN', 'ar', 'id', 'tr', 'vi'
        ]
        transcript = None
        found_lang = None
        transcript_type = ""
        # ... (rest of transcript fetching logic - no changes needed here) ...
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
        st.error(f"An unexpected error occurred fetching transcript: {e}")
        return None


def format_transcript_for_llm(transcript_data):
    # ... (same code using segment.start, segment.text and dot notation) ...
    formatted = []
    if not transcript_data: return ""
    for segment in transcript_data:
        try:
            time_val = segment.start
            text_val = segment.text
        except AttributeError:
             try:
                 time_val = segment['start']
                 text_val = segment['text']
             except (KeyError, TypeError):
                 st.warning(f"Skipping segment due to unexpected format: {segment}")
                 continue
        time_str = format_time(time_val)
        text = str(text_val or '').replace('\n', ' ').strip()
        formatted.append(f"[{time_str}] {text}")
    return "\n".join(formatted)


# --- Modified for Azure ---
def get_topics_with_llm(client: AzureOpenAI, azure_deployment_name: str, transcript_string: str):
    """Sends transcript to the Azure OpenAI LLM, shows status in Streamlit."""
    if not client:
        st.error("Azure OpenAI client not initialized.")
        return None
    if not transcript_string:
        st.warning("Transcript string is empty. Cannot send to LLM.")
        return None
    if not azure_deployment_name:
        st.error("Azure deployment name not provided.")
        return None

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
        # Use the Azure deployment name
        st.write(f"Sending transcript to Azure deployment '{azure_deployment_name}' for topic identification...")

        response = client.chat.completions.create(
            model=azure_deployment_name, # Azure uses deployment name here
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=800
        )

        if response.choices and response.choices[0].message and response.choices[0].message.content:
             content = response.choices[0].message.content
             finish_reason = response.choices[0].finish_reason
             if finish_reason == 'content_filter':
                 st.warning("Azure OpenAI response flagged by content filters.")
             elif finish_reason == 'length':
                 st.warning(f"Azure OpenAI response may be truncated due to max_tokens limit.")
             return content
        else:
            st.error("Received an empty or incomplete response from Azure OpenAI.")
            st.json(response)
            return None

    # Catch specific OpenAI/Azure errors
    except openai.AuthenticationError:
        st.error("Azure OpenAI Authentication Error: Invalid API Key or Endpoint.")
        return None
    except openai.RateLimitError:
        st.error("Azure OpenAI Rate Limit Exceeded. Please wait or check your Azure quota/deployment limits.")
        return None
    except openai.APITimeoutError:
        st.error("Azure OpenAI API request timed out.")
        return None
    except openai.NotFoundError:
        st.error(f"Azure deployment '{azure_deployment_name}' not found. Check deployment name, endpoint, and API version.")
        return None
    except openai.APIConnectionError as e:
         st.error(f"Azure OpenAI API Connection Error: Could not connect to endpoint. Check network/endpoint URL. Details: {e}")
         return None
    except Exception as e:
        # Catch other potential errors
        st.error(f"An unexpected error occurred during the Azure OpenAI API call: {e}")
        return None

def parse_llm_response(response_text):
    # ... (same code using st.warning/st.caption for parsing issues) ...
    if not response_text: return []
    topics = []
    pattern = re.compile(r"^\s*(?:\*?\*?)(.*?)(?:\*?\*?)\s*:\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*$", re.MULTILINE)
    matches = pattern.findall(response_text)
    if not matches:
        st.warning("Could not parse LLM response using primary regex. Attempting simple line split.")
        lines = response_text.strip().split('\n')
        fallback_found = False
        for line in lines:
            line = line.strip()
            if not line: continue
            if ':' in line:
                parts = line.split(':', 1)
                topic = parts[0].strip().strip('*').strip()
                timestamp = parts[1].strip()
                if re.match(r"^\d{1,2}:\d{2}(?::\d{2})?$", timestamp):
                    topics.append({"topic": topic, "timestamp": timestamp})
                    fallback_found = True
                else: st.caption(f"Skipping line (invalid timestamp format): {line}")
            else: st.caption(f"Skipping line (no colon found): {line}")
        if not fallback_found: st.error("Failed to parse topics using fallback method.")
    else:
        for match in matches:
            topic_title = match[0].strip().strip('*').strip()
            topics.append({"topic": topic_title, "timestamp": match[1].strip()})
    return topics


# --- Streamlit App ---

st.set_page_config(page_title="YouTube Topic Extractor (Azure)", layout="wide")
st.title("🎥 YouTube Video Topic Extractor (Azure OpenAI)")
st.caption("Enter a YouTube URL to get timestamped topics using Azure OpenAI GPT-4o.")

# --- Azure OpenAI Configuration (Prioritize Secrets) ---
st.sidebar.header("Azure OpenAI Configuration")

azure_endpoint = None
azure_api_key = None
azure_api_version = None
azure_deployment_name = None # This is the deployment name for gpt-4o
client = None # Initialize client as None

# Attempt to load from Streamlit secrets first
try:
    if hasattr(st, 'secrets'):
        azure_endpoint = st.secrets.get("AZURE_OPENAI_ENDPOINT")
        azure_api_key = st.secrets.get("AZURE_OPENAI_KEY")
        azure_api_version = st.secrets.get("AZURE_OPENAI_API_VERSION")
        azure_deployment_name = st.secrets.get("AZURE_OPENAI_DEPLOYMENT_NAME") # e.g., "gpt-4o" or your custom name

        if all([azure_endpoint, azure_api_key, azure_api_version, azure_deployment_name]):
            st.sidebar.info("Using Azure OpenAI credentials from Secrets.")
        else:
             # Clear partial secrets if not all are found
             azure_endpoint = azure_api_key = azure_api_version = azure_deployment_name = None
             st.sidebar.warning("Partial Azure credentials in Secrets. Please provide missing values below or complete Secrets.")

except Exception as e:
    # Handle case where secrets might not be available/accessible
    st.sidebar.caption("Could not access Streamlit Secrets.")
    pass

# Fallback to manual input if secrets are missing or incomplete
if not all([azure_endpoint, azure_api_key, azure_api_version, azure_deployment_name]):
    st.sidebar.subheader("Enter Azure Details (if not using Secrets):")

    azure_endpoint_input = st.sidebar.text_input(
        "Azure OpenAI Endpoint:",
        value=azure_endpoint or "https://YOUR_RESOURCE_NAME.openai.azure.com/", # Provide example format
        help="Your Azure OpenAI resource endpoint URL."
    )
    # Use the potentially incorrect key from user as default, but warn them
    azure_api_key_input = st.sidebar.text_input(
        "Azure OpenAI Key:",
        type="password",
        value=azure_api_key or "", # Don't default the key here for security
        help="Your Azure OpenAI API Key. Looks short? Double-check it."
    )
    # Default to a likely API version, allow override
    azure_api_version_input = st.sidebar.text_input(
        "Azure API Version:",
        value=azure_api_version or "2024-05-01-preview", # Default to a recent version likely supporting GPT-4o
        help="e.g., 2024-05-01-preview or 2024-02-01"
    )
    azure_deployment_name_input = st.sidebar.text_input(
        "GPT-4o Deployment Name:",
        value=azure_deployment_name or "gpt-4o", # Default guess, user MUST verify
        help="The name you gave your GPT-4o model when deploying it in Azure."
    )

    # Use input values only if secrets weren't fully loaded
    if not azure_endpoint: azure_endpoint = azure_endpoint_input
    if not azure_api_key: azure_api_key = azure_api_key_input
    if not azure_api_version: azure_api_version = azure_api_version_input
    if not azure_deployment_name: azure_deployment_name = azure_deployment_name_input


# --- YouTube URL Input ---
youtube_url = st.text_input("Enter the YouTube video URL:", placeholder="e.g., https://www.youtube.com/watch?v=...")

# --- Analyze Button ---
# Disable button if any required Azure cred or URL is missing
all_creds_provided = all([azure_endpoint, azure_api_key, azure_api_version, azure_deployment_name])
button_disabled = not youtube_url or not all_creds_provided
analyze_button = st.button("Analyze Video", disabled=button_disabled, type="primary")

# --- User Guidance Messages ---
if not youtube_url and not all_creds_provided:
    st.info("Please provide Azure OpenAI credentials (via Secrets or input) and enter a YouTube URL.")
elif not all_creds_provided:
    st.info("Please provide all Azure OpenAI credentials in the sidebar (or configure Secrets).")
elif not youtube_url:
    st.info("Please enter a YouTube URL.")


# --- Processing Logic ---
if analyze_button and all_creds_provided:
    # Initialize AzureOpenAI Client
    try:
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=azure_api_version,
        )
        # Perform a lightweight test call (e.g., list models, though not strictly necessary)
        # client.models.list() # Optional: Test connectivity - might incur small cost/quota usage
        st.sidebar.success("Azure OpenAI client ready.")

    except openai.AuthenticationError:
         st.error("Azure Authentication Error: Invalid API Key or Endpoint. Please check your credentials.")
         client = None
         st.stop()
    except Exception as e:
         st.error(f"Error initializing Azure OpenAI client: {e}")
         client = None
         st.stop()

    # Proceed only if client initialization was successful
    if client:
        analysis_successful = False
        parsed_topics_result = None
        llm_response_result = None
        error_message = None

        video_id = get_video_id(youtube_url)

        if not video_id:
            st.error("Invalid YouTube URL or could not extract Video ID.")
        else:
            st.subheader(f"Processing Video ID: `{video_id}`")

            with st.status("Analyzing video via Azure OpenAI...", expanded=True) as status:
                # 1. Get Transcript
                st.write("Fetching transcript...")
                transcript_data = get_transcript(video_id)

                if transcript_data:
                    # 2. Format Transcript
                    st.write("Formatting transcript...")
                    formatted_transcript = format_transcript_for_llm(transcript_data)

                    if formatted_transcript:
                        # 3. Get Topics from Azure LLM
                        # Pass the client and deployment name
                        llm_response = get_topics_with_llm(client, azure_deployment_name, formatted_transcript)
                        llm_response_result = llm_response # Store for potential later display

                        if llm_response:
                            # 4. Parse LLM Response
                            st.write("Parsing topics from response...")
                            parsed_topics = parse_llm_response(llm_response)

                            if parsed_topics:
                                status.update(label="Analysis Complete!", state="complete", expanded=False)
                                parsed_topics_result = parsed_topics
                                analysis_successful = True
                            else:
                                status.update(label="Parsing Failed", state="error", expanded=True)
                                error_message = "Could not parse topics from the LLM response."
                        else:
                            status.update(label="LLM Analysis Failed", state="error", expanded=False)
                            error_message = "Failed to get response from Azure LLM."
                    else:
                        status.update(label="Formatting Failed", state="error", expanded=False)
                        error_message = "Formatted transcript is empty."
                else:
                    status.update(label="Transcript Fetch Failed", state="error", expanded=False)
                    error_message = "Could not retrieve transcript."

        # --- Display Results (Outside Status Block) ---
        if analysis_successful and parsed_topics_result:
            st.subheader("✅ Identified Topics and Timestamps (Azure GPT-4o)")
            for item in parsed_topics_result:
                st.markdown(f"- **{item['topic']}**: `{item['timestamp']}`")

            if llm_response_result:
                 with st.expander("View Raw LLM Response"):
                    st.text_area("Raw Response", llm_response_result, height=200, disabled=True)

        elif error_message:
            st.error(f"Analysis failed: {error_message}")
            if llm_response_result and "Could not parse topics" in error_message:
                 st.subheader("Raw LLM Response (Parsing Failed):")
                 st.text_area("Raw Response", llm_response_result, height=200, disabled=True)

# --- End of App ---