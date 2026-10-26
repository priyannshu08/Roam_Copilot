import os
from dotenv import load_dotenv
load_dotenv()
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"


import streamlit as st
import time
import requests
import smtplib
import base64
import re
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Any, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import json
from io import BytesIO
from markdown_pdf import MarkdownPdf, Section
import tempfile
import hashlib
import copy
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from queue import Queue, Empty
from free_tier import FreeTierClient, QuotaUnavailable

# Helper function to get API keys safely
def get_api_key(key):
    # Try env var first
    val = os.getenv(key)
    if val:
        return val
    # Try st.secrets
    try:
        return st.secrets.get(key, "")
    except:
        return ""

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from operator import add

from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import Annotated
from langchain_community.utilities import WikipediaAPIWrapper
from tavily import TavilyClient
import googlemaps

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch



# Set up page config
st.set_page_config(
    page_title="AI Trip Planner",
    page_icon="✈️",
    layout="wide"
)

# Wikipedia Image Function (Enhanced)
def get_wikipedia_image(query: str, delay: float = 0.5) -> Optional[str]:
    """Fetch a cached destination image from Tavily."""
    try:
        data = search_travel(query, max_results=1, include_images=True)
        if data.get("images"):
            image_data = data["images"][0]
            if isinstance(image_data, str):
                return image_data
            return image_data.get("url")
    except Exception:
        return None
    return None

# Initialize APIs
@st.cache_resource
def initialize_apis():
    """Initialize all API clients"""
    try:
        llm = FreeTierClient(ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            google_api_key=get_api_key("GOOGLE_API_KEY"),
            temperature=0.7,
            max_retries=0,
            timeout=90,
            max_output_tokens=4096
        ))
        
        tavily_client = TavilyClient(api_key=get_api_key("TAVILY_API_KEY"))
        wikipedia = WikipediaAPIWrapper()
        
        return llm, tavily_client, wikipedia
    except Exception as e:
        st.error(f"Error initializing APIs: {str(e)}")
        return None, None, None

# Enhanced state structure with Annotated fields for parallel updates
class TripPlannerState(TypedDict):
    origin: str
    destination: str
    start_date: str
    end_date: str
    budget: int
    num_travelers: int
    interests: List[str]
    
    # Agent outputs with Annotated for parallel updates
    destination_info: Dict[str, Any]
    destination_image: Optional[str]
    places_info: List[Dict[str, Any]]
    flight_options: List[Dict[str, Any]]
    hotel_options: List[Dict[str, Any]]
    activities: List[Dict[str, Any]]
    weather_info: Dict[str, Any]
    final_itinerary: str
    itinerary_error: str
    
    # Progress tracking - using Annotated for parallel updates
    current_step: Annotated[List[str], add]
    error_messages: Annotated[List[str], add]
    progress: int

# Initialize APIs
llm, tavily_client, wikipedia = initialize_apis()

@st.cache_data(ttl=3600, max_entries=256, show_spinner=False)
def search_travel(query, search_depth="basic", max_results=3, include_images=False):
    return tavily_client.search(
        query=query, search_depth=search_depth, max_results=max_results,
        include_images=include_images
    )


def service_error(error, service):
    if isinstance(error, QuotaUnavailable):
        return str(error)
    status = getattr(error, "status_code", None)
    message = str(error).lower()
    if status in (429, 432, 433) or any(term in message for term in ("quota", "rate limit", "usage limit", "credit")):
        return f"{service}'s free request, token, or credit allowance is exhausted. Retry after its quota resets. Partial research is saved."
    if status in (401, 403) or any(term in message for term in ("api_key_invalid", "api key not valid", "unauthorized")):
        return f"{service} rejected its API key. Check the local credentials."
    return f"{service} request failed ({type(error).__name__}). Partial research is saved; retry later."


def research_agent(state: TripPlannerState) -> TripPlannerState:
    """Research destination and get main image"""
    try:
        destination = state["destination"]
        # st.write(f"🔍 Researching {destination}...") # Removed to prevent UI clutter
        
        results = search_travel(f"{destination} travel tourism practical information")
        destination_image = get_wikipedia_image(f"{destination} landmark")
        destination_info = {
            "description": "\n".join(result.get("content", "")[:500] for result in results.get("results", [])[:3]),
            "sources": [result.get("url", "") for result in results.get("results", [])[:3]]
        }
        
        return {
            "destination_info": destination_info,
            "destination_image": destination_image,
            "current_step": ["research_complete"],
            "progress": 20,
        }
    except Exception as e:
        return {
            "destination_info": {},
            "destination_image": None,
            "error_messages": [service_error(e, "Tavily research")],
        }


def places_agent(state: TripPlannerState) -> dict:
    """Agent to fetch top places using Tavily"""
    try:
        destination = state["destination"]
        # st.write(f"📍 Finding places for {destination}...") # Removed to prevent UI clutter

        places_query = f"top tourist attractions in {destination}"
        places_result = search_travel(
            query=places_query,
            search_depth="basic",
            max_results=6
        )

        places_info = []
        for r in places_result.get("results", [])[:6]:
            place_details = {
                "name": r.get("title", ""),
                "rating": None,   # Tavily doesn’t give ratings directly
                "address": "",
                "types": [],
                "place_id": None,
                "image_url": None
            }
            places_info.append(place_details)

        return {
            "places_info": places_info,
            "current_step": ["places_complete"],
            "progress": 40
        }

    except Exception as e:
        return {
            "places_info": [],
            "error_messages": [service_error(e, "Tavily places")]
        }


def weather_agent(state: TripPlannerState) -> dict:
    """Agent to fetch weather info (mock for now, later OpenWeatherMap)"""
    try:
        destination = state["destination"]
        # st.write(f"⛅ Checking weather for {destination}...") # Removed to prevent UI clutter

        # For now, just search with Tavily (acts like knowledge/weather lookup)
        weather_query = f"current weather and climate in {destination}"
        weather_result = search_travel(
            query=weather_query,
            search_depth="basic",
            max_results=3
        )

        # Very rough extraction — can refine with LLM later
        weather_info = {
            "forecast": "\n".join(result.get("content", "")[:400] for result in weather_result.get("results", [])[:2]),
            "temperature": "Not verified",
            "conditions": "Search summaries only; check a forecast near departure"
        }

        return {
            "weather_info": weather_info,
            "current_step": ["weather_complete"],
            "progress": 50
        }

    except Exception as e:
        return {
            "weather_info": {},
            "error_messages": [service_error(e, "Tavily weather")],
            "current_step": ["weather_complete"]
        }



def activities_agent(state: TripPlannerState) -> TripPlannerState:
    """Find activities and events using Tavily"""
    try:
        destination = state["destination"]
        interests = state["interests"]
        start_date = state["start_date"]
        
        # st.write(f"🎯 Finding activities in {destination}...") # Removed to prevent UI clutter
        
        activities = []
        for interest in interests[:3]:
            query = f"{interest} activities events {destination} {start_date} things to do experiences"
            results = search_travel(
                query=query,
                search_depth="basic",
                max_results=3
            )
            
            for result in results.get("results", []):
                activity = {
                    "title": result.get("title", ""),
                    "description": result.get("content", "")[:200],
                    "category": interest,
                    "url": result.get("url", ""),
                    "image_url": None
                }
                activities.append(activity)
        
        return {
            "activities": activities[:8],
            "current_step": ["activities_complete"],
            "progress": 90,

        }
    except Exception as e:
        return {
            "activities": [],
            "error_messages": [service_error(e, "Tavily activities")],
        }

def itinerary_agent(state: TripPlannerState) -> TripPlannerState:
    """Generate final comprehensive itinerary"""
    try:
        # st.write("📝 Generating your personalized itinerary...") # Removed to prevent UI clutter
        
        # Calculate trip duration
        start = datetime.strptime(state["start_date"], "%Y-%m-%d")
        end = datetime.strptime(state["end_date"], "%Y-%m-%d")
        duration = (end - start).days
        
        prompt = f"""
        Create a detailed {duration}-day trip itinerary for {state["destination"]}.
        
        TRIP DETAILS:
        - Departure city: {state["origin"]}
        - Destination: {state["destination"]}
        - Dates: {state["start_date"]} to {state["end_date"]} ({duration} days)
        - Budget: ${state["budget"]} for {state["num_travelers"]} travelers
        - Interests: {", ".join(state["interests"])}
        
        AVAILABLE DATA:
        
        Destination Info: {json.dumps(state.get("destination_info", {}))}
        
        Top Places: {json.dumps([p.get("name", "") + " (Rating: " + str(p.get("rating", "N/A")) + ")" for p in state.get("places_info", [])[:5]])}
        
        Flight Options: {json.dumps(state.get("flight_options", []))}
        
        Hotel Options: {json.dumps(state.get("hotel_options", []))}
        
        Activities Available: {json.dumps([a.get("title", "") + " - " + a.get("category", "") for a in state.get("activities", [])[:6]])}
        
        Weather: {json.dumps(state.get("weather_info", {}))}

        Search results are unverified source summaries, not confirmed bookings or live prices.
        Cite their source URLs. Do not invent prices, availability, ratings, or weather.
        Label any budget estimates explicitly and note missing research.
        
        CREATE A COMPREHENSIVE ITINERARY WITH:
        
        ## 🗓️ TRIP OVERVIEW
        - Destination summary
        - Duration and dates
        - Budget breakdown
        
        ## ✈️ FLIGHTS & ARRIVAL
        - Recommended flight options (from the search results)
        - Airport transfer tips
        
        ## 🏨 ACCOMMODATION
        - Top 3 hotel recommendations (from search results with prices)
        - Area recommendations
        
        ## 📅 DAY-BY-DAY ITINERARY
        For each day, include:
        - Morning activity (specific place/attraction)
        - Afternoon activity
        - Evening suggestion
        - Estimated costs
        - Transportation tips
        
        ## 🎯 MUST-DO ACTIVITIES
        - Top attractions (from places found)
        - Experience recommendations
        - Booking tips
        
        ## 💰 BUDGET BREAKDOWN
        - Flights: estimated cost
        - Hotels: per night cost
        - Activities: daily budget
        - Food: daily budget
        - Transport: local transport
        
        ## 📋 PRACTICAL TIPS
        - Local customs
        - Transportation
        - Safety tips
        - Emergency contacts
        
        Make it engaging, practical, and well-formatted with emojis and clear sections.
        """
        
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))

        return {
            "final_itinerary": content,
            "itinerary_error": "",
            "current_step": ["complete"],
            "progress": 100,
        }
    except Exception as e:
        return {
            "final_itinerary": "",
            "itinerary_error": service_error(e, "Gemini"),
        }

# Conditional routing function
def should_continue_planning(state: TripPlannerState) -> str:
    """Decide next step based on research results"""
    destination_info = state.get("destination_info", {})
    
    # Check if we have basic destination info
    if destination_info and not destination_info.get("error"):
        return "continue"
    else:
        return "error"

# PDF Generation Function
def generate_pdf(itinerary_text: str, destination: str, state) -> BytesIO:
    """
    Generate PDF from Markdown text using markdown_pdf library.
    Input:
        itinerary_text: str -> the Markdown itinerary content
        destination: str -> destination name
        state: TripPlannerState or dict (kept for compatibility)
    Returns:
        BytesIO buffer containing the PDF
    """
    pdf_buffer = BytesIO()

    # Clean text to remove emojis and non-BMP characters which break reportlab/markdown_pdf
    # This regex removes characters outside the Basic Multilingual Plane (BMP)
    clean_itinerary = re.sub(r'[^\u0000-\uFFFF]', '', itinerary_text)
    
    if not clean_itinerary.strip():
        clean_itinerary = "Itinerary content could not be rendered due to character encoding issues. Please view the text version."

    # Create MarkdownPdf object
    pdf = MarkdownPdf(toc_level=0)

    # Optional: add a title section with destination
    title_md = f"# Trip Itinerary: {destination}\n\n"
    
    # Combine title and itinerary text
    full_md = title_md + clean_itinerary

    # Add as a section
    pdf.add_section(Section(full_md))

    # Save PDF to BytesIO
    pdf.save(pdf_buffer)
    pdf_buffer.seek(0)

    return pdf_buffer
    



# Email function
def send_email(recipient_email: str, pdf_buffer: BytesIO, destination: str):
    """Send itinerary PDF via email using Gmail SMTP"""
    try:
        # Reset buffer position
        pdf_buffer.seek(0)

        sender_email = get_api_key("GMAIL_USER")
        sender_password = get_api_key("GMAIL_APP_PASSWORD")
        
        if not sender_email or not sender_password:
            st.error("Email credentials not configured in secrets!")
            return False
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"Your Trip Itinerary for {destination}"
        
        body = f"""
        Hello!
        
        Your personalized trip itinerary for {destination} is ready!
        
        Please find the detailed PDF itinerary attached.
        
        Happy travels!
        
        Best regards,
        AI Trip Planner
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach PDF
        pdf_buffer.seek(0)
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(pdf_buffer.read())
        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename= "{destination.replace(" ", "_")}_itinerary.pdf"',
        )
        msg.attach(part)
        
        # Send email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())


        
        return True
        
    except Exception as e:
        st.error(f"Email error: {str(e)}")
        return False

def flights_agent(state: TripPlannerState) -> TripPlannerState:
    """Search flights using Tavily"""
    try:
        origin = state.get("origin", "India") # Default fallback if missing
        destination = state["destination"]
        start_date = state["start_date"]
        budget = state["budget"]
        
        # Search for detailed flight information
        flight_query = f"flights to {destination} from {origin} {start_date} prices airlines booking"
        flight_results = search_travel(
            query=flight_query,
            search_depth="basic",
            max_results=3
        )
        flight_options = [
            {"title": result.get("title", "Flight source"),
             "description": result.get("content", "")[:500],
             "url": result.get("url", "")}
            for result in flight_results.get("results", [])[:3]
        ]
        
        # Return only flight-specific updates
        return {
            "flight_options": flight_options,
            "current_step": ["flights_complete"],
        }
    except Exception as e:
        return {
            "flight_options": [],
            "error_messages": [service_error(e, "Tavily flights")],
            "current_step": ["flights_complete"]
        }

def hotels_agent(state: TripPlannerState) -> TripPlannerState:
    """Search hotels using Tavily"""
    try:
        destination = state["destination"]
        budget = state["budget"]
        
        # Search for hotel information
        hotel_query = f"best hotels {destination} accommodation booking prices reviews {budget} budget"
        hotel_results = search_travel(
            query=hotel_query,
            search_depth="basic",
            max_results=3
        )
        hotel_options = [
            {"name": result.get("title", "Hotel source"),
             "description": result.get("content", "")[:500],
             "url": result.get("url", "")}
            for result in hotel_results.get("results", [])[:3]
        ]
        
        # Return only hotel-specific updates
        return {
            "hotel_options": hotel_options,
            "current_step": ["hotels_complete"],
        }
        
    except Exception as e:
        return {
            "hotel_options": [],
            "error_messages": [service_error(e, "Tavily hotels")],
            "current_step": ["hotels_complete"]
        }

# Create enhanced workflow with proper parallel execution
def create_workflow():
    workflow = StateGraph(TripPlannerState)
    
    # Add nodes
    workflow.add_node("research", research_agent)
    workflow.add_node("places", places_agent)
    workflow.add_node("weather", weather_agent)
    workflow.add_node("flights", flights_agent)
    workflow.add_node("hotels", hotels_agent)
    workflow.add_node("activities", activities_agent)
    workflow.add_node("itinerary", itinerary_agent)

     # Gate node to wait for all parallel branches
    workflow.add_node("sync_gate", lambda s: s) #barrier code


    # Define the flow
    workflow.set_entry_point("research")
    
    # Sequential flow: research -> places
    workflow.add_edge("research", "places")
    
    # Parallel execution: places -> [flights, hotels, weather]
    workflow.add_edge("places", "flights")
    workflow.add_edge("places", "hotels")
    workflow.add_edge("places", "weather")

    # Wait for all parallel agents before activities
    workflow.add_edge("flights", "sync_gate")
    workflow.add_edge("hotels", "sync_gate")
    workflow.add_edge("weather", "sync_gate")


    # Barrier condition: proceed only when ALL parallel branches are done
    def _barrier_condition(state: TripPlannerState) -> str:
        steps = set(state.get("current_step", []))
        required = {"flights_complete", "hotels_complete", "weather_complete"}
        return "go" if required.issubset(steps) else "wait"


    workflow.add_conditional_edges(
        "sync_gate",
        _barrier_condition,
        {
            "go": "activities",   # move forward once all are done
            "wait": END,   # temporary halt until other branch completes
        },
    )

  
    # Final steps
    workflow.add_edge("activities", "itinerary")
    workflow.add_edge("itinerary", END)
    
    return workflow.compile()

def trip_cache_key(details):
    fields = ("origin", "destination", "start_date", "end_date", "budget", "num_travelers", "interests")
    values = {field: details.get(field) for field in fields}
    values["interests"] = sorted(values["interests"] or [])
    encoded = json.dumps(values, sort_keys=True, default=str).encode("utf-8")
    return "trip_result_" + hashlib.sha256(encoded).hexdigest()


def get_cached_trip(details):
    state = st.session_state.get(trip_cache_key(details))
    if state and time.time() - state.get("_saved_at", 0) < 3600:
        return copy.deepcopy(state)
    return None


def invoke_with_status(prompt):
    notice = st.empty()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(llm.invoke, prompt)
            while True:
                try:
                    return future.result(timeout=0.25)
                except FutureTimeout:
                    if future.done():
                        return future.result()
                    notice.info(llm.get_status() or "Waiting for Gemini")
    finally:
        notice.empty()


def stream_trip(initial_state):
    saved = get_cached_trip(initial_state)
    if saved and saved.get("final_itinerary"):
        yield {"itinerary": saved}
        return
    resume = bool(saved and saved.get("_research_complete"))
    state = copy.deepcopy(saved if resume else initial_state)
    events = Queue()

    def produce():
        try:
            if resume:
                events.put(("update", {"itinerary": itinerary_agent(state)}))
            else:
                for output in create_workflow().stream(initial_state):
                    events.put(("update", output))
        except Exception as error:
            events.put(("error", error))
        finally:
            events.put(("done", None))

    notice = st.empty()
    notice.info("Retrying itinerary from saved research" if resume else "Collecting travel research")
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(produce)
            while True:
                try:
                    kind, output = events.get(timeout=0.25)
                except Empty:
                    notice.info(llm.get_status() or "Collecting travel research")
                    continue
                if kind == "done":
                    break
                if kind == "error":
                    raise RuntimeError(service_error(output, "Trip planning")) from None
                for node, updates in output.items():
                    if node == "sync_gate":
                        continue
                    for field, value in updates.items():
                        if field in ("current_step", "error_messages"):
                            state[field] = list(dict.fromkeys(state.get(field, []) + value))
                        else:
                            state[field] = value
                    if node == "itinerary":
                        state["_research_complete"] = True
                        st.session_state.pop(f"pdf_buffer_{trip_cache_key(initial_state)}", None)
                    state["_saved_at"] = time.time()
                    st.session_state[trip_cache_key(initial_state)] = copy.deepcopy(state)
                yield output
    finally:
        notice.empty()


def render_partial_research(state, key="partial_research"):
    st.warning(state.get("itinerary_error") or "The itinerary is incomplete. Available research has been saved.")
    for message in dict.fromkeys(state.get("error_messages", [])):
        st.warning(message)
    fields = ("destination_info", "places_info", "flight_options", "hotel_options", "weather_info", "activities")
    research = {field: state.get(field) for field in fields if state.get(field)}
    with st.expander("Saved travel research", expanded=True):
        st.json(research)
    st.download_button("Download research", json.dumps(research, indent=2),
                       file_name="travel_research.json", mime="application/json", key=key)


# Enhanced Streamlit UI
def main():
    st.title("🌍 AI Multi-Agent Trip Planner")
    
    # Sidebar Navigation
    page = st.sidebar.radio("Choose Mode", ["📝 Trip Planner (Form)", "💬 Travel Chatbot"])
    
    # Sidebar Configuration (Shared)
    with st.sidebar:
        st.header("🔑 Configuration")
        
        if llm and tavily_client:
            st.info("API clients configured; credentials and quota are checked when used.")
        else:
            st.error("❌ Check API keys in secrets")
            st.code("""
            # Add to .streamlit/secrets.toml:
            GOOGLE_API_KEY = "your_key"
            TAVILY_API_KEY = "your_key"
            GMAIL_USER = "your_gmail@gmail.com"  # Optional for email
            GMAIL_APP_PASSWORD = "your_app_password"  # Optional for email
            """)
        
        st.markdown("---")
        st.markdown("### 🔄 Workflow")
        st.markdown("""
        1. **Research** destination info
        2. **Places & Weather** (parallel)
        3. **Flights & Hotels** (parallel)
        4. **Activities** based on interests
        5. **Generate** final itinerary
        """)
        st.markdown("---")
        if st.button("🗑️ Clear Trip Cache", help="Clear cached results to force new planning"):
            # Clear trip-related session state
            keys_to_clear = [k for k in st.session_state.keys() if k.startswith(('trip_', 'email_sent_', 'pdf_buffer', 'chat_history'))]
            for key in keys_to_clear:
                del st.session_state[key]
            search_travel.clear()
            if llm:
                llm.clear_cache()
            st.rerun()

    if page == "📝 Trip Planner (Form)":
        render_form_planner()
    else:
        render_chat_planner()

def render_chat_planner():
    st.header("💬 Chat with your AI Travel Team")
    st.markdown("I'll help you plan your trip step-by-step. Let's start!")

    # Initialize session state for the conversation
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        # Initial greeting
        greeting = "Hello! I'm your Travel Coordinator. To get started, where would you like to go?"
        st.session_state.chat_history.append({
            "role": "assistant", 
            "name": "Coordinator",
            "avatar": "👩‍💼",
            "content": greeting
        })
    
    if "trip_details" not in st.session_state:
        st.session_state.trip_details = {
            "origin": None,
            "destination": None,
            "start_date": None,
            "end_date": None,
            "budget": None,
            "num_travelers": None,
            "interests": [],
            "is_complete": False
        }

    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            if message.get("name"):
                st.write(f"**{message['name']}**")
            st.markdown(message["content"])

    # Chat input
    saved_trip = get_cached_trip(st.session_state.trip_details)
    if saved_trip and not saved_trip.get("final_itinerary"):
        render_partial_research(saved_trip, key="chat_saved_research")
        if st.button("Retry itinerary", key="chat_retry_itinerary"):
            try:
                for output in stream_trip(saved_trip):
                    pass
                retried = get_cached_trip(saved_trip)
                if retried and retried.get("final_itinerary"):
                    st.session_state.chat_history.append({
                        "role": "assistant", "name": "Itinerary Agent",
                        "content": retried["final_itinerary"]
                    })
                st.rerun()
            except Exception as error:
                st.error(service_error(error, "Trip planning"))

    if prompt := st.chat_input("Type your answer here..."):
        # 1. Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Process input to update trip details (Allow modifications even if complete)
        with st.chat_message("assistant", avatar="👩‍💼"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🤔 Thinking...")

            needs_revision = False

            try:
                # LLM Prompt to extract info and decide next question
                current_details = st.session_state.trip_details
                
                # Get recent chat history to provide context
                recent_history = st.session_state.chat_history[-3:] if len(st.session_state.chat_history) > 0 else []
                
                extraction_prompt = f"""
                You are a friendly travel agent. 
                Current known trip details: {json.dumps(current_details)}
                Recent conversation history: {json.dumps(recent_history)}
                User just said: "{prompt}"
                
                Task:
                1. Update the trip details based on user input AND the context of the recent conversation (e.g. if the bot asked "Where are you flying from?", the user's answer is the origin).
                2. Check what is missing (Required: origin, destination, start_date, end_date, budget, num_travelers, interests).
                3. If details are missing, formulate the NEXT friendly question to ask the user.
                4. If all details are present, set 'is_complete' to true.
                5. Set 'needs_revision' to true IF:
                   - The user explicitly asks to CHANGE any details.
                   - The user confirms the details (e.g. "yes", "correct", "go ahead", "proceed", "make plan").
                   - The user just provided the last missing detail.
                6. Set 'needs_revision' to false ONLY IF:
                   - The user is just chatting (e.g. "thanks", "cool") without changing anything.
                   - The user is asking a question about the destination without changing plans.
                
                Return ONLY a JSON object:
                {{
                    "updated_details": {{...all fields...}},
                    "response_message": "Your friendly response or next question here",
                    "needs_revision": true/false
                }}
                """
                
                # Call LLM
                response = invoke_with_status(extraction_prompt)
                content = getattr(response, "content", str(response))
                
                # Parse JSON
                try:
                    clean_json = content.replace("```json", "").replace("```", "").strip()
                    result = json.loads(clean_json)
                    
                    # Update state
                    st.session_state.trip_details = result["updated_details"]
                    bot_response = result["response_message"]
                    needs_revision = result.get("needs_revision", False)
                    
                except Exception as e:
                    bot_response = "I didn't quite catch that. Could you clarify?"
                
                # Show Bot Response
                message_placeholder.markdown(bot_response)
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "name": "Coordinator",
                    "avatar": "👩‍💼",
                    "content": bot_response
                })

                # 4. If complete and needs revision, Trigger the Agents!
                if st.session_state.trip_details.get("is_complete") and needs_revision:
                    
                    # Prepare State
                    details = st.session_state.trip_details
                    initial_state = TripPlannerState(
                        origin=details['origin'],
                        destination=details['destination'],
                        start_date=details['start_date'],
                        end_date=details['end_date'],
                        budget=details['budget'],
                        num_travelers=details['num_travelers'],
                        interests=details['interests'],
                        destination_info={},
                        destination_image=None,
                        places_info=[],
                        flight_options=[],
                        hotel_options=[],
                        activities=[],
                        weather_info={},
                        final_itinerary="",
                        current_step=["starting"],
                        error_messages=[],
                        progress=0
                    )

                    # Run Workflow Stream
                    
                    # Define Agent Personas
                    agent_personas = {
                        "research": {"name": "Research Agent", "avatar": "🕵️", "msg": "I'm researching the destination history and culture..."},
                        "places": {"name": "Places Agent", "avatar": "📍", "msg": "I'm looking for the top tourist attractions..."},
                        "weather": {"name": "Weather Agent", "avatar": "⛅", "msg": "Checking the forecast for your dates..."},
                        "flights": {"name": "Flight Agent", "avatar": "✈️", "msg": "Searching for the best flight options..."},
                        "hotels": {"name": "Hotel Agent", "avatar": "🏨", "msg": "Finding accommodation within your budget..."},
                        "activities": {"name": "Activity Agent", "avatar": "🎯", "msg": "Looking for local events and activities based on your interests..."},
                        "itinerary": {"name": "Itinerary Agent", "avatar": "📝", "msg": "Compiling everything into your final itinerary..."}
                    }

                    # Stream the execution
                    with st.container():
                        st.write("🚀 **Starting your trip planning mission!**")
                        
                        # Status container for logs
                        status = st.status("Working on your trip...", expanded=True)
                        
                        # Dashboard for live results
                        dashboard = st.container()
                        
                        final_itinerary_text = ""
                        
                        for output in stream_trip(initial_state):
                            for node_name, node_state in output.items():
                                if node_name in agent_personas:
                                    persona = agent_personas[node_name]
                                    
                                    # Update Status Log
                                    status.write(f"**{persona['avatar']} {persona['name']}**: {persona['msg']}")
                                    
                                    # Agent Internals (New)
                                    with status:
                                        with st.expander(f"🛠️ {persona['name']} Internal Logs", expanded=False):
                                            st.write(f"**Task:** {persona['msg']}")
                                            st.write("**Agent Output:**")
                                            # Create a clean view of the output
                                            clean_state = {k:v for k,v in node_state.items() if k not in ['current_step', 'progress', 'error_messages']}
                                            # Truncate long strings for readability
                                            if "final_itinerary" in clean_state:
                                                clean_state["final_itinerary"] = clean_state["final_itinerary"][:200] + "... [Full content in final output]"
                                            st.json(clean_state)
                                            if 'error_messages' in node_state and node_state['error_messages']:
                                                st.error(f"Errors: {node_state['error_messages']}")

                                    # Update Dashboard with Rich Content
                                    with dashboard:
                                        if node_name == "research":
                                            info = node_state.get("destination_info", {})
                                            img = node_state.get("destination_image")
                                            col1, col2 = st.columns([1, 2])
                                            if img:
                                                col1.image(img, use_container_width=True)
                                            if info:
                                                col2.info(f"**Destination Insight**: {info.get('description', '')[:200]}...")
                                                
                                        elif node_name == "weather":
                                            weather = node_state.get("weather_info", {})
                                            temp = weather.get('temperature', 'N/A')
                                            cond = weather.get('conditions', 'Unavailable')
                                            st.success(f"🌤️ **Weather**: {temp} - {cond}")
                                            
                                        elif node_name == "places":
                                            places = node_state.get("places_info", [])
                                            st.write(f"📍 Found {len(places)} attractions")
                                            
                                        elif node_name == "flights":
                                            flights = node_state.get("flight_options", [])
                                            st.write(f"✈️ Found {len(flights)} flight options")

                                        elif node_name == "itinerary":
                                            final_itinerary_text = node_state.get("final_itinerary", "")
                                            status.update(
                                                label="Planning complete" if final_itinerary_text else "Research saved; itinerary unavailable",
                                                state="complete" if final_itinerary_text else "error",
                                                expanded=not bool(final_itinerary_text)
                                            )

                    # Final Itinerary Display
                    if final_itinerary_text:
                        with st.container():
                            st.markdown("### 🏁 Your Trip Itinerary")
                            st.markdown(final_itinerary_text)
                            
                            # PDF Download
                            pdf_buffer = generate_pdf(final_itinerary_text, details['destination'], {})
                            st.download_button(
                                label="📄 Download PDF Itinerary",
                                data=pdf_buffer,
                                file_name=f"{details['destination']}_itinerary.pdf",
                                mime="application/pdf"
                            )
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "name": "Itinerary Agent",
                            "avatar": "📝",
                            "content": final_itinerary_text
                        })
                    else:
                        saved = get_cached_trip(initial_state)
                        if saved:
                            render_partial_research(saved)

            except Exception as e:
                st.error(service_error(e, "Gemini"))

def render_form_planner():
    st.markdown("Plan your perfect trip with AI-powered research, real-time search, and beautiful itineraries!")
    
    # Main form
    with st.form("enhanced_trip_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            origin = st.text_input("🛫 Departure City", placeholder="e.g., New York, London")
            destination = st.text_input("🏙️ Destination", placeholder="e.g., Tokyo, Japan")
            start_date = st.date_input("📅 Start Date", datetime.now() + timedelta(days=7))
            budget = st.number_input("💰 Budget (USD)", min_value=500, max_value=50000, value=3000, step=100)
        
        with col2:
            end_date = st.date_input("📅 End Date", datetime.now() + timedelta(days=14))
            num_travelers = st.number_input("👥 Travelers", min_value=1, max_value=10, value=2)
            interests = st.multiselect(
                "🎯 Your Interests",
                ["Culture", "Food", "Adventure", "Museums", "Nature", "Shopping", "Nightlife", "History", "Art", "Music", "Architecture", "Local Experiences"],
                default=["Culture", "Food", "Local Experiences"]
            )
        
        
        submitted = st.form_submit_button("🚀 Plan My Amazing Trip!", use_container_width=True)
    
    details = dict(origin=origin, destination=destination, start_date=str(start_date),
                   end_date=str(end_date), budget=budget, num_travelers=num_travelers,
                   interests=interests)
    trip_key = trip_cache_key(details) if destination else None
    saved_state = get_cached_trip(details) if destination else None

    if destination and llm and (submitted or saved_state):
        # Validation
        if not origin:
            st.error("Please enter a departure city!")
            return

        if start_date >= end_date:
            st.error("End date must be after start date!")
            return
        
        if not interests:
            st.error("Please select at least one interest!")
            return
        
        
        # Add this check before running the workflow:
        if saved_state and (saved_state.get("final_itinerary") or not submitted):
            # Use cached results
            final_state = saved_state
            if final_state.get("final_itinerary"):
                st.info("Using saved trip plan")
        else:
            # Run workflow and cache results
            with st.spinner("Planning your trip..."):
                # Initialize state with proper list format for Annotated fields
                initial_state = TripPlannerState(
                    origin=origin,
                    destination=destination,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    budget=budget,
                    num_travelers=num_travelers,
                    interests=interests,
                    destination_info={},
                    destination_image=None,
                    places_info=[],
                    flight_options=[],
                    hotel_options=[],
                    activities=[],
                    weather_info={},
                    final_itinerary="",
                    current_step=["starting"],  # Initialize as list
                    error_messages=[],  # Initialize as empty list
                    progress=0
                )

            
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_container = st.container()
                
                try:
                    for output in stream_trip(initial_state):
                        final_state = get_cached_trip(initial_state)
                        if final_state:
                            progress_bar.progress(final_state.get("progress", 0))
                
                except Exception as e:
                    st.error(service_error(e, "Trip planning"))
                    saved = get_cached_trip(initial_state)
                    if saved:
                        render_partial_research(saved)
                    return

        if trip_key and trip_key in st.session_state:
            final_state = st.session_state[trip_key]
            if not final_state.get("final_itinerary"):
                render_partial_research(final_state)
                return
            for message in dict.fromkeys(final_state.get("error_messages", [])):
                st.warning(message)
            
            # Get current form values for email functionality
            form_data = st.session_state.get('form_data', {})

            # Display results
            if final_state.get("final_itinerary"):
                # Show destination image if available
                if final_state.get("destination_image"):
                    st.image(final_state["destination_image"], caption=f"{destination}", use_column_width=True)
                
                st.markdown("## 📋 Your Complete Trip Itinerary")
                st.markdown(final_state["final_itinerary"])
            
                # Detailed sections
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    with st.expander("🔍 Destination Details"):
                        dest_info = final_state.get("destination_info", {})
                        if dest_info:
                            st.json(dest_info)
                
                with col2:
                    with st.expander("📍 Top Places & Attractions"):
                        places = final_state.get("places_info", [])
                        for i, place in enumerate(places[:5], 1):
                            st.write(f"**{i}. {place.get('name', 'Unknown')}**")
                            st.write(f"⭐ Rating: {place.get('rating', 'N/A')}")
                            st.write(f"📍 {place.get('address', 'N/A')}")
                            if place.get('image_url'):
                                st.image(place['image_url'], width=200)
                            st.write("---")
                
                with col3:
                    with st.expander("✈️ Flight Options Found"):
                        flights = final_state.get("flight_options", [])
                        if flights:
                            for i, flight in enumerate(flights, 1):
                                st.write(f"**Option {i}:**")
                                st.write(flight.get("title", "Flight source"))
                                st.write(flight.get("description", ""))
                                if flight.get("url"):
                                    st.link_button("Source", flight["url"])
                                st.write("---")
                        else:
                            st.info("No specific flight options extracted. Check the main itinerary for flight guidance.")
                
                # Hotel section
                st.markdown("### 🏨 Hotel Search Results")
                hotels = final_state.get("hotel_options", [])
                if hotels:
                    hotel_cols = st.columns(min(len(hotels), 3))
                    for i, hotel in enumerate(hotels[:3]):
                        with hotel_cols[i]:
                            st.write(f"**{hotel.get('name', 'Hotel')}**")
                            st.write(hotel.get("description", ""))
                            if hotel.get("url"):
                                st.link_button("Source", hotel["url"])
                else:
                    st.info("Hotel recommendations are included in the main itinerary above.")
                
                # Activities section
                st.markdown("### 🎯 Recommended Activities")
                activities = final_state.get("activities", [])
                if activities:
                    activity_cols = st.columns(2)
                    for i, activity in enumerate(activities[:6]):
                        with activity_cols[i % 2]:
                            st.write(f"**{activity.get('title', 'Activity')}**")
                            st.write(f"🎯 Category: {activity.get('category', 'N/A')}")
                            st.write(f"📝 {activity.get('description', 'No description')}")
                            if activity.get('image_url'):
                                st.image(activity['image_url'], width=150)
                            st.write("---")
                # Generate PDF buffer once

                
                # Initialize session_state flags
                show_email_key = f"show_email_{trip_key}"
                email_key = f"email_sent_{trip_key}"

                if show_email_key not in st.session_state:
                    st.session_state[show_email_key] = False
                if email_key not in st.session_state:
                    st.session_state[email_key] = False

                if f"pdf_buffer_{trip_key}" not in st.session_state:
                    st.session_state[f"pdf_buffer_{trip_key}"] = generate_pdf(
                        final_state["final_itinerary"], 
                        destination, 
                        final_state
                    )
                
            
                pdf_buffer = st.session_state[f"pdf_buffer_{trip_key}"]
                
                
                # Download and Email options
                st.markdown("### 📥 Get Your Itinerary")
                
                col_download, col_email = st.columns(2)
                
                with col_download:
                    # Generate and offer PDF download
                    
                    st.download_button(
                        label="📄 Download PDF Itinerary",
                        data=pdf_buffer.getvalue(),
                        file_name=f"{destination.replace(' ', '_')}_itinerary.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

                                  
                            
                    # Text download fallback
                    st.download_button(
                        label="📝 Download Text Version",
                        data=final_state["final_itinerary"],
                        file_name=f"{destination.replace(' ', '_')}_itinerary.txt",
                        mime="text/plain"
                    )
                    
                    
            
        
    elif submitted and not llm:
        st.error("❌ Please configure your API keys first!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    🤖 Powered by LangGraph Multi-Agent System | Built with Streamlit<br>
    Uses: Gemini LLM • Tavily Search • Google Places • Wikipedia
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()