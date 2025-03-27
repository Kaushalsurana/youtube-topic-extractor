import streamlit as st
# Keep standard openai import for error types like AuthenticationError
import openai
# ADD Azure Client import
from openai import AzureOpenAI
import os
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import time

# --- Helper Functions (Keep format_time, get_video_id, get_transcript, format_transcript_for_llm - NO CHANGES NEEDED) ---

def format_time(seconds):
    """Converts seconds to HH:MM:SS or MM:SS format."""
    if seconds is None: return "00:00"
    try: seconds = float(seconds)
    except (ValueError, TypeError): return "??:??"
    seconds = int(seconds)
    hours = seconds // 3600; minutes = (seconds % 3600) // 60; secs = seconds % 60
    if hours > 0: return f"{hours:02}:{minutes:02}:{secs:02}"
    else: return f"{minutes:02}:{secs:02}"

def get_video_id(url):
    """Extracts YouTube video ID from various URL formats."""
    if not isinstance(url, str): return None
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^?]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^?]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([^?]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url);
        if match: return match.group(1)
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url): return url
    return None

def get_transcript(video_id):
    """Fetches the transcript, displaying status in Streamlit."""
    # --- This function remains exactly the same ---
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        preferred_languages = ['en', 'en-US', 'en-GB', 'hi', 'es', 'fr', 'de', 'pt', 'it', 'ru', 'ja', 'ko', 'zh-Hans', 'zh-CN', 'ar', 'id', 'tr', 'vi']
        transcript = None; found_lang = None; transcript_type = ""
        for lang in preferred_languages: # Try manual
            try: transcript = transcript_list.find_manually_created_transcript([lang]); found_lang = lang; transcript_type = "manually created"; st.info(f"Found {transcript_type} transcript ({found_lang})."); break
            except NoTranscriptFound: continue
        if not transcript: # Try generated
            for lang in preferred_languages:
                try: transcript = transcript_list.find_generated_transcript([lang]); found_lang = lang; transcript_type = "auto-generated"; st.info(f"Found {transcript_type} transcript ({found_lang})."); break
                except NoTranscriptFound: continue
        if not transcript: # Handle not found
            try:
                available_langs = [f"{t.language} ({'manual' if t.is_translatable else 'generated' if t.is_generated else 'unknown'})" for t in transcript_list]
                if available_langs: st.warning(f"No transcript found in preferred languages."); st.caption(f"Available: {', '.join(available_langs)}")
                else: st.error(f"No transcripts found for video ID: {video_id}")
                return None
            except Exception as e: st.error(f"Error listing transcripts: {e}"); return None
        st.info(f"Fetching transcript content for {found_lang} ({transcript_type})..."); transcript_data = transcript.fetch(); return transcript_data
    except TranscriptsDisabled: st.error(f"Transcripts disabled for video ID: {video_id}"); return None
    except Exception as e: st.error(f"Unexpected error fetching transcript: {e}"); return None

def format_transcript_for_llm(transcript_data):
    """Formats transcript data into a string suitable for LLM processing."""
     # --- This function remains exactly the same ---
    formatted = [];
    if not transcript_data: return ""
    for segment in transcript_data:
        try: time_val = segment.start; text_val = segment.text
        except AttributeError:
             try: time_val = segment['start']; text_val = segment['text']
             except (KeyError, TypeError): st.warning(f"Skipping segment: {segment}"); continue
        time_str = format_time(time_val); text = str(text_val or '').replace('\n', ' ').strip(); formatted.append(f"[{time_str}] {text}")
    return "\n".join(formatted)


# --- MODIFIED FOR AZURE ---
# Takes Azure client and deployment name now
def get_topics_with_llm(client: AzureOpenAI, azure_deployment_name: str, transcript_string: str):
    """Sends transcript to the Azure OpenAI LLM, shows status in Streamlit."""
    # --- Checks remain similar ---
    if not client: st.error("Azure OpenAI client not initialized."); return None
    if not transcript_string: st.warning("Transcript string is empty."); return None
    if not azure_deployment_name: st.error("Azure deployment name not provided."); return None

    # --- USE THE PROMPTS FROM YOUR ORIGINAL FILE ---
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
    # --- END OF USING ORIGINAL PROMPTS ---

    try:
        # Use the Azure deployment name passed into the function
        st.write(f"Sending transcript to Azure deployment '{azure_deployment_name}' (GPT-4o) for topic identification...")

        # --- API CALL USING AZURE DEPLOYMENT NAME ---
        response = client.chat.completions.create(
            model=azure_deployment_name, # Use deployment name here
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=800
        )
        # --- END OF API CALL CHANGE ---

        # Response checking remains similar
        if response.choices and response.choices[0].message and response.choices[0].message.content:
             content = response.choices[0].message.content; finish_reason = response.choices[0].finish_reason
             if finish_reason == 'content_filter': st.warning("Azure response flagged by content filters.")
             elif finish_reason == 'length': st.warning(f"Azure response may be truncated.")
             return content
        else: st.error("Empty/incomplete response from Azure."); st.json(response); return None

    # --- Update Error Handling for Azure Context ---
    except openai.AuthenticationError: st.error("Azure Authentication Error: Invalid API Key or Endpoint."); return None
    except openai.RateLimitError: st.error("Azure Rate Limit Exceeded."); return None
    except openai.APITimeoutError: st.error("Azure API request timed out."); return None
    except openai.NotFoundError: st.error(f"Azure deployment '{azure_deployment_name}' not found or API version mismatch."); return None
    except openai.APIConnectionError as e: st.error(f"Azure Connection Error: Check Endpoint URL. Details: {e}"); return None
    except Exception as e: st.error(f"Unexpected Azure API error: {e}"); return None
    # --- END OF ERROR HANDLING UPDATE ---

def parse_llm_response(response_text):
    """Parses the LLM response to extract topics and timestamps."""
    # --- This function remains exactly the same ---
    if not response_text: return []
    topics = []; pattern = re.compile(r"^\s*(?:\*?\*?)(.*?)(?:\*?\*?)\s*:\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*$", re.MULTILINE)
    matches = pattern.findall(response_text)
    if not matches:
        st.warning("Could not parse using regex. Trying line split."); lines = response_text.strip().split('\n'); fallback_found = False
        for line in lines:
            line = line.strip();
            if not line: continue
            if ':' in line:
                parts = line.split(':', 1); topic = parts[0].strip().strip('*').strip(); timestamp = parts[1].strip()
                if re.match(r"^\d{1,2}:\d{2}(?::\d{2})?$", timestamp): topics.append({"topic": topic, "timestamp": timestamp}); fallback_found = True
                else: st.caption(f"Skip invalid timestamp: {line}")
            else: st.caption(f"Skip no colon: {line}")
        if not fallback_found: st.error("Failed fallback parse.")
    else:
        for match in matches: topics.append({"topic": match[0].strip().strip('*').strip(), "timestamp": match[1].strip()})
    return topics


# --- Streamlit App ---
st.set_page_config(page_title="YouTube Topic Extractor (Azure)", layout="wide")
st.title("🎥 YouTube Video Topic Extractor (Azure OpenAI)") # Updated Title
st.caption("Enter a YouTube URL to get timestamped topics (using Azure GPT-4o).") # Updated Caption

# --- Azure OpenAI Configuration (Prioritize Secrets) ---
st.sidebar.header("Azure OpenAI Configuration")

azure_endpoint = None
azure_api_key = None
azure_api_version = None
azure_deployment_name = None # Deployment name for gpt-4o
client = None # Initialize client as None
secrets_found = False

# Attempt to load from Streamlit secrets first
try:
    if hasattr(st, 'secrets'):
        # Use .get() for safer retrieval, returns None if key doesn't exist
        azure_endpoint = st.secrets.get("AZURE_OPENAI_ENDPOINT")
        azure_api_key = st.secrets.get("AZURE_OPENAI_KEY")
        azure_api_version = st.secrets.get("AZURE_OPENAI_API_VERSION")
        azure_deployment_name = st.secrets.get("AZURE_OPENAI_DEPLOYMENT_NAME") # e.g., "gpt-4o"

        # Check if ALL secrets were successfully retrieved
        if all([azure_endpoint, azure_api_key, azure_api_version, azure_deployment_name]):
            st.sidebar.info("Using Azure OpenAI credentials from Secrets.")
            secrets_found = True
        else:
             # Indicate if only some secrets are found, prompting for the rest
             st.sidebar.warning("Partial/Missing Azure credentials in Secrets. Provide missing values below.")
             # Ensure partially loaded secrets don't prevent fallback inputs
             if not azure_endpoint: azure_endpoint = None
             if not azure_api_key: azure_api_key = None
             if not azure_api_version: azure_api_version = None
             if not azure_deployment_name: azure_deployment_name = None

except Exception as e:
    st.sidebar.caption(f"Could not access Streamlit Secrets: {e}")
    pass # Proceed to allow manual input


# Fallback to manual input if secrets are not fully configured
if not secrets_found:
    st.sidebar.subheader("Enter Azure Details (if not using Secrets):")

    # Use placeholders or reasonable defaults for inputs
    azure_endpoint_input = st.sidebar.text_input(
        "Azure OpenAI Endpoint:",
        value=azure_endpoint or "", # Use empty string if secret wasn't found
        placeholder="https://YOUR_RESOURCE.openai.azure.com/",
        help="Your Azure OpenAI resource endpoint URL."
    )
    azure_api_key_input = st.sidebar.text_input(
        "Azure OpenAI Key:",
        type="password",
        value="", # Never default password/key inputs
        help="Your Azure OpenAI API Key."
    )
    azure_api_version_input = st.sidebar.text_input(
        "Azure API Version:",
        value=azure_api_version or "2024-05-01-preview", # Default to a likely valid version
        help="e.g., 2024-05-01-preview"
    )
    azure_deployment_name_input = st.sidebar.text_input(
        "GPT-4o Deployment Name:",
        value=azure_deployment_name or "gpt-4o", # Default guess
        help="The exact name for your GPT-4o deployment in Azure."
    )

    # Assign input values ONLY if secrets weren't found for that specific value
    if not azure_endpoint: azure_endpoint = azure_endpoint_input
    if not azure_api_key: azure_api_key = azure_api_key_input
    if not azure_api_version: azure_api_version = azure_api_version_input
    if not azure_deployment_name: azure_deployment_name = azure_deployment_name_input


# --- YouTube URL Input (Remains the same) ---
youtube_url = st.text_input("Enter the YouTube video URL:", placeholder="e.g., https://www.youtube.com/watch?v=...")

# --- Analyze Button (Condition updated for Azure creds) ---
# Check if all required credentials have a value (either from secrets or input)
all_creds_provided = all([azure_endpoint, azure_api_key, azure_api_version, azure_deployment_name])
button_disabled = not youtube_url or not all_creds_provided
analyze_button = st.button("Analyze Video", disabled=button_disabled, type="primary")

# --- User Guidance Messages (Updated for Azure creds) ---
if not youtube_url and not all_creds_provided:
    st.info("Please provide Azure OpenAI credentials (via Secrets or input) and enter a YouTube URL.")
elif not all_creds_provided:
    st.info("Please provide all Azure OpenAI credentials in the sidebar (or configure Secrets).")
elif not youtube_url:
    st.info("Please enter a YouTube URL.")


# --- Processing Logic (Updated for Azure Client Initialization) ---
if analyze_button and all_creds_provided: # Check if button clicked AND creds available
    client = None # Initialize client as None before try block
    # --- Initialize AzureOpenAI Client using final credential values ---
    try:
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=azure_api_version,
        )
        # Optional: Test connectivity if needed
        # client.models.list()
        st.sidebar.success("Azure OpenAI client configured.") # Use success if init works
    except openai.AuthenticationError:
         st.error("Azure Authentication Error! Check API Key/Endpoint/Version.")
         client = None; st.stop() # Stop if client fails
    except Exception as e:
         st.error(f"Error initializing Azure client: {e}")
         client = None; st.stop() # Stop if client fails
    # --- END OF AZURE CLIENT INITIALIZATION ---

    # Proceed only if client initialization was successful
    if client:
        analysis_successful = False; parsed_topics_result = None; llm_response_result = None; error_message = None # Var initialization

        video_id = get_video_id(youtube_url) # Remains same

        if not video_id:
            st.error("Invalid YouTube URL or could not extract Video ID.")
        else:
            st.subheader(f"Processing Video ID: `{video_id}`")
            with st.status("Analyzing video via Azure OpenAI...", expanded=True) as status: # Updated status message
                # 1. Get Transcript (Remains same)
                st.write("Fetching transcript..."); transcript_data = get_transcript(video_id)
                if transcript_data:
                    # 2. Format Transcript (Remains same)
                    st.write("Formatting transcript..."); formatted_transcript = format_transcript_for_llm(transcript_data)
                    if formatted_transcript:
                        # --- 3. Call AZURE LLM function ---
                        # Pass client and deployment name (from secrets or input)
                        llm_response = get_topics_with_llm(client, azure_deployment_name, formatted_transcript)
                        # --- END AZURE LLM CALL ---
                        llm_response_result = llm_response # Store result
                        if llm_response:
                            # 4. Parse LLM Response (Remains same)
                            st.write("Parsing topics..."); parsed_topics = parse_llm_response(llm_response)
                            if parsed_topics:
                                status.update(label="Analysis Complete!", state="complete", expanded=False); parsed_topics_result = parsed_topics; analysis_successful = True
                            else: status.update(label="Parsing Failed", state="error", expanded=True); error_message = "Could not parse topics."
                        else: status.update(label="LLM Failed", state="error", expanded=False); error_message = "No response from Azure LLM."
                    else: status.update(label="Formatting Failed", state="error", expanded=False); error_message = "Formatted transcript empty."
                else: status.update(label="Transcript Failed", state="error", expanded=False); error_message = "Could not get transcript."

        # --- Display Results (Remains the same, maybe update subheader text) ---
        if analysis_successful and parsed_topics_result:
            st.subheader("✅ Identified Topics and Timestamps (Azure GPT-4o)") # Updated subheader
            for item in parsed_topics_result: st.markdown(f"- **{item['topic']}**: `{item['timestamp']}`")
            if llm_response_result:
                 with st.expander("View Raw LLM Response"): st.text_area("Raw Response", llm_response_result, height=200, disabled=True)
        elif error_message:
            st.error(f"Analysis failed: {error_message}")
            if llm_response_result and "Could not parse topics" in error_message:
                 st.subheader("Raw LLM Response (Parsing Failed):"); st.text_area("Raw Response", llm_response_result, height=200, disabled=True)

# --- End of App ---