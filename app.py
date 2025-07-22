import os
from datetime import datetime
from openai import OpenAI
import yaml
import streamlit as st
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
import httpx
from typing import Dict, Optional, Tuple

import system_prompts
from ollama_manager import OllamaManager
from toolbox_core import ToolboxSyncClient

# Page configuration
st.set_page_config(
    page_title="OpenShift Partner Labs",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load configuration
def load_config() -> Dict:
    """
    Load configuration from the config.yaml file.

    Returns:
        Dict: Configuration dictionary
    """
    with open("config.yaml", "r") as file:
        config_file = yaml.safe_load(file)

    return config_file

config = load_config()

# Set up OAuth flow
def create_oauth_flow() -> Flow:
    """
    Create Google OAuth flow.

    Returns:
        Flow: Google OAuth flow
    """
    # Ensure redirect URI is properly formatted
    redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8501")

    # Create client configuration
    client_config = {
        "web": {
            "client_id": os.environ.get("GOOGLE_CLIENT_ID", ""),
            "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET", ""),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri]
        }
    }

    # Create and configure the flow
    flow = Flow.from_client_config(
        client_config=client_config,
        scopes=["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"],
        redirect_uri=redirect_uri
    )

    return flow

# Get user info from Google
def get_user_info(credentials: Credentials) -> Dict:
    """
    Get user information from Google.

    Args:
        credentials (Credentials): Google OAuth credentials

    Returns:
        Dict: User information
    """
    response = httpx.get(
        "https://www.googleapis.com/oauth2/v1/userinfo",
        headers={"Authorization": f"Bearer {credentials.token}"}
    )
    return response.json()

# Check if the user is authorized
def is_authorized(email: str) -> bool:
    """
    Check if the user email is in the preauthorized list or has an authorized domain.

    Args:
        email (str): User email

    Returns:
        bool: True if authorized, False otherwise
    """
    # Allow if email is in the preauthorized list or if the list is empty
    if email in config["preauthorized"]["emails"] or not config["preauthorized"]["emails"]:
        return True

    # Allow redhat.com email addresses
    if email.lower().endswith("@redhat.com"):
        return True

    return False

# Handle OAuth flow
def handle_oauth() -> Tuple[bool, Optional[Dict]]:
    """
    Handle OAuth flow and return authentication status and user info.

    Returns:
        Tuple[bool, Optional[Dict]]: Authentication status and user info
    """
    # Check if we have a code in the URL (callback from Google)
    query_params = st.query_params

    if "code" in query_params:
        try:
            flow = create_oauth_flow()
            # Get the authorization code from query parameters
            # Handle both list and string formats
            _code = query_params["code"]
            if isinstance(_code, list):
                _code = _code

            # Exchange the authorization code for credentials
            flow.fetch_token(
                code=_code
            )
            credentials = flow.credentials

            # Get user info
            user_info = get_user_info(credentials)

            # Check if the user is authorized
            if is_authorized(user_info.get("email", "")):
                # Store user info in the session state
                st.session_state["authenticated"] = True
                st.session_state["user_info"] = user_info

                # Clear the URL parameters
                st.query_params.clear()

                return True, user_info
            else:
                st.error(f"Email {user_info.get('email')} is not authorized to access this application.")
                st.session_state["authenticated"] = False
                return False, None

        except Exception as e:
            error_message = str(e)
            st.error(f"Authentication error: {error_message}")

            # Provide more specific guidance based on the error
            if "invalid_grant" in error_message and "Malformed auth code" in error_message:
                st.info("This error often occurs when the authorization code is not properly formatted or has expired. Please try logging in again.")
            elif "redirect_uri_mismatch" in error_message:
                st.info("The redirect URI in your request doesn't match the one registered in the Google Cloud Console. Please check your configuration.")
            elif "invalid_client" in error_message:
                st.info("The client ID or client secret is incorrect. Please check your credentials in the config.yaml file.")
            elif "multiple values for keyword argument 'redirect_uri'" in error_message:
                st.info("There was an issue with the OAuth configuration. Please try logging in again.")

            st.session_state["authenticated"] = False
            return False, None

    return st.session_state.get("authenticated", False), st.session_state.get("user_info", None)

def get_system_prompt(user_prompt: str = None, persona: str = None) -> str:
    return system_prompts.default_persona

def init_session_state() -> None:
    # Initialize empty message history for storing chat conversations
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Set the session ID using UUID
    if 'session_id' not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())

    # Initialize ollama manager
    if 'ollama' not in st.session_state:
        st.session_state.ollama = None

    if 'vllm' not in st.session_state:
        st.session_state.vllm = None

def initialize_ollama():
    try:
        # Create a new instance of the ollama wrapper class
        st.session_state.ollama = OllamaManager(
            host=config["ollama"]["host"],
            model=config["ollama"]["chat_model"],
            options=dict(config["ollama"]["options"])
        )

        return True

    except Exception as e:
        # Display user-friendly error message in the Streamlit interface
        st.error(f"Failed to initialize Ollama: {str(e)}")
        return False

def get_ollama_response(prompt: str) -> str:
    try:
        messages = []

        for message in st.session_state.messages:
            messages.append({"role": message["role"], "content": message["content"]})

        messages.append({"role": "user", "content": prompt})

        response = st.session_state.ollama.chat(messages)

        return response.message.content
    except Exception as e:
        # Return the error message
        return f"❌ Error: {str(e)}"

def initialize_vllm():
    try:
        openai_api_key = config["vllm_config"]["api_key"]
        schema = "https://" if config["vllm_config"]["secure"] else "http://"
        openai_api_base = f"{schema}{config['vllm_config']['chat_model']}-{config['vllm_config']['namespace']}.{config['vllm_config']['base_url']}"

        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        st.session_state.vllm = client

        return True
    except Exception as e:
        st.error(f"Failed to initialize vLLM: {str(e)}")
        return False

def get_vllm_response(prompt: str) -> str:
    try:
        messages = []

        for message in st.session_state.messages:
            messages.append({"role": message["role"], "content": message["content"]})

        messages.append({"role": "user", "content": prompt})

        response = st.session_state.vllm.chat.completions.create(
            model=config["vllm_config"]["chat_model"],
            messages=messages
        )

        return response.choices[0].message.content
    except Exception as e:
        # Return the error message
        return f"❌ Error: {str(e)}"

def use_toolbox_tool(tool_name: str, tool_params: Dict) -> str:
    toolbox = ToolboxSyncClient("http://localhost:5000")
    tool = toolbox.load_tool(tool_name)
    result = tool(**tool_params)
    return result


# Main application
def main():
    user_info = {}

    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not config["credentials"]["enabled"]:
        st.session_state["authenticated"] = True
    else:
        # Check authentication
        authenticated, user_info = handle_oauth()

    # If not authenticated, show the login page
    if not st.session_state["authenticated"]:
        st.title("OpenShift Partner Labs")
        st.write("Please log in with your Google account to continue.")

        # Create OAuth flow
        flow = create_oauth_flow()
        auth_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent"
        )

        # Login button
        st.markdown(f"<a href='{auth_url}' target='_self'><button style='background-color:#4285F4;color:white;border:none;padding:10px 20px;border-radius:4px;cursor:pointer;'>Login with Google</button></a>", unsafe_allow_html=True)

    # If authenticated, show the main content
    else:
        # Initialize all session state variables
        init_session_state()

        # === SIDEBAR CONFIGURATION ===
        with st.sidebar:
            if not st.session_state.ollama and config["ollama"]["enabled"]:
                initialize_ollama()
            else:
                initialize_vllm()

            st.subheader(f"Welcome, {user_info.get('name', 'User')}!")

            # Display user info
            st.image(user_info.get("picture", "https://i.pravatar.cc/150"))
            st.write(f"Email: {user_info.get('email', 'no-reply@redhat.com')}")

            # Display chat statistics
            st.divider()
            st.subheader("📊 Chat Statistics")
            st.metric("Total Messages", len(st.session_state.messages))

            # Model and session information
            st.divider()

            if not config["ollama"]["enabled"]:
                st.caption(f"Model: {config['vllm_config']['chat_model']}")
            else:
                st.caption(f"Model: {config['ollama']['chat_model']}")
            st.caption(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

            # Connection status indicator
            if st.session_state.ollama or st.session_state.vllm:
                st.caption("🟢 Connected")
            else:
                st.caption("🔴 Disconnected")

            # Logout button
            if st.button("Logout"):
                st.session_state["authenticated"] = False
                st.session_state["user_info"] = None
                st.rerun()

        # Main app content
        st.header("OpenShift Partner Labs")

        # === MAIN CHAT INTERFACE ===
        # Display all previous chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input handling
        # The walrus operator := captures the input while checking if it exists
        if user_prompt := st.chat_input("Type your message here..."):
            # Immediately display the user message
            with st.chat_message("user"):
                st.markdown(user_prompt)

            # Add the user message to state
            st.session_state.messages.append({"role": "user", "content": user_prompt})

            # TODO: get_system_prompt should send the user_prompt to a LLM with the intent
            #  of selecting a system prompt based on the user prompt.
            system_prompt = get_system_prompt(user_prompt)

            # Get a response from the model
            with st.spinner("Thinking..."):
                if config["ollama"]["enabled"]:
                    response = get_ollama_response(system_prompt + user_prompt + "\n</user>")
                else:
                    response = get_vllm_response(system_prompt + user_prompt + "\n</user>")

            # Add the model response to state
            st.session_state.messages.append({"role": "assistant", "content": response})
            print(response)

            # Reload streamlit
            st.rerun()


if __name__ == "__main__":
    main()