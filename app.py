import streamlit as st
from transformers import pipeline
import torch
import json
from datetime import datetime
import time
import hashlib
from typing import List, Dict, Optional
import csv
import io

# Page configuration
st.set_page_config(
    page_title="Prompt Optimization Tool",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .optimized-prompt {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 20px 0;
    }
    .stButton > button {
        width: 100%;
    }
    .copy-button {
        background-color: #28a745;
        color: white;
    }
    .header-title {
        color: #1f77b4;
        text-align: center;
        padding: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'current_prompt' not in st.session_state:
        st.session_state.current_prompt = None
    if 'clarifying_questions' not in st.session_state:
        st.session_state.clarifying_questions = None
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    if 'optimized_prompt' not in st.session_state:
        st.session_state.optimized_prompt = None
    if 'step' not in st.session_state:
        st.session_state.step = 'input'
    if 'processing_times' not in st.session_state:
        st.session_state.processing_times = []
    if 'total_prompts' not in st.session_state:
        st.session_state.total_prompts = 0

@st.cache_resource
def load_model():
    """Load and cache the LLM model"""
    try:
        # Try to use GPU if available
        device = 0 if torch.cuda.is_available() else -1
        
        # Using Mistral 7B as default (you can switch to Llama)
        # Note: For production, you might want to use smaller models for faster inference
        model = pipeline(
            'text-generation',
            model='mistralai/Mistral-7B-Instruct-v0.1',
            device=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            max_length=1024
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        # Fallback to a smaller model if the main one fails
        try:
            model = pipeline(
                'text-generation',
                model='microsoft/phi-2',
                device=-1,
                max_length=1024
            )
            return model
        except:
            return None

def generate_clarifying_questions(prompt: str, model) -> List[str]:
    """Generate clarifying questions based on the user's prompt"""
    
    clarification_prompt = f"""You are an expert prompt analyst. Analyze this user prompt and generate 3-7 essential clarifying questions that will help optimize it.

User Prompt: {prompt}

Focus on identifying:
- Missing context or background
- Vague requirements
- Undefined scope or constraints
- Implicit assumptions
- Output format preferences

Generate only the questions, numbered 1-7, no other text.
"""
    
    try:
        with st.spinner("Generating clarifying questions..."):
            start_time = time.time()
            
            # Generate response
            response = model(
                clarification_prompt,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
            
            end_time = time.time()
            st.session_state.processing_times.append(end_time - start_time)
            
            # Extract questions from response
            text = response[0]['generated_text'].replace(clarification_prompt, '').strip()
            
            # Parse numbered questions
            questions = []
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Clean up the question
                    question = line.lstrip('0123456789.-) ').strip()
                    if question:
                        questions.append(question)
            
            # Ensure we have at least 3 questions
            if len(questions) < 3:
                default_questions = [
                    "What is the primary goal or objective?",
                    "Who is the target audience?",
                    "What format should the output be in?",
                    "Are there any specific constraints or requirements?",
                    "What level of detail is needed?"
                ]
                questions.extend(default_questions[:max(3, 5-len(questions))])
            
            return questions[:7]  # Limit to 7 questions
            
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        # Return default questions as fallback
        return [
            "What is the primary goal or objective?",
            "Who is the target audience?",
            "What format should the output be in?",
            "Are there any specific constraints or requirements?",
            "What level of detail is needed?"
        ]

def optimize_prompt(original_prompt: str, answers: Dict[str, str], model) -> str:
    """Generate optimized prompt based on original and clarification answers"""
    
    # Format answers for the prompt
    formatted_answers = "\n".join([f"Q: {q}\nA: {a}" for q, a in answers.items() if a])
    
    optimization_prompt = f"""You are an expert prompt engineer. Rewrite this prompt to be maximally effective for an AI assistant.

Original Prompt: {original_prompt}

Clarification Answers:
{formatted_answers}

Create an optimized prompt that:
- Front-loads critical context
- Uses precise, actionable language
- Includes clear success criteria
- Specifies all constraints
- Eliminates ambiguity

Output only the optimized prompt, no explanations.
"""
    
    try:
        with st.spinner("Optimizing your prompt..."):
            start_time = time.time()
            
            # Generate optimized prompt
            response = model(
                optimization_prompt,
                max_new_tokens=400,
                temperature=0.3,
                do_sample=True,
                top_p=0.9
            )
            
            end_time = time.time()
            st.session_state.processing_times.append(end_time - start_time)
            
            # Extract optimized prompt
            text = response[0]['generated_text'].replace(optimization_prompt, '').strip()
            
            # Clean up the response
            text = text.strip()
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            
            return text
            
    except Exception as e:
        st.error(f"Error optimizing prompt: {str(e)}")
        # Create a basic optimization as fallback
        context_parts = [original_prompt]
        for q, a in answers.items():
            if a:
                context_parts.append(f"{q}: {a}")
        return "\n".join(context_parts)

def save_to_history(original: str, optimized: str, questions: List[str], answers: Dict[str, str]):
    """Save prompt optimization to session history"""
    entry = {
        'timestamp': datetime.now().isoformat(),
        'original': original,
        'optimized': optimized,
        'questions': questions,
        'answers': answers,
        'id': hashlib.md5(f"{original}{datetime.now()}".encode()).hexdigest()[:8]
    }
    
    st.session_state.history.insert(0, entry)
    
    # Keep only last 20 entries
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[:20]
    
    st.session_state.total_prompts += 1

def reset_workflow():
    """Reset the workflow to start fresh"""
    st.session_state.current_prompt = None
    st.session_state.clarifying_questions = None
    st.session_state.user_answers = {}
    st.session_state.optimized_prompt = None
    st.session_state.step = 'input'

def display_sidebar():
    """Display the sidebar with history and admin features"""
    with st.sidebar:
        st.header("📝 Prompt History")
        
        # Check for admin mode
        query_params = st.query_params
        is_admin = query_params.get('admin') == ['true']
        
        if is_admin:
            st.info("🔐 Admin Mode Active")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Prompts", st.session_state.total_prompts)
            with col2:
                avg_time = sum(st.session_state.processing_times) / len(st.session_state.processing_times) if st.session_state.processing_times else 0
                st.metric("Avg Time (s)", f"{avg_time:.2f}")
            
            if st.button("📥 Export History (CSV)"):
                export_history_csv()
        
        if not st.session_state.history:
            st.info("No prompts optimized yet. Start by entering a prompt!")
        else:
            for entry in st.session_state.history:
                timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%H:%M:%S")
                preview = entry['original'][:50] + "..." if len(entry['original']) > 50 else entry['original']
                
                with st.expander(f"🕐 {timestamp} - {preview}"):
                    st.subheader("Original Prompt")
                    st.text(entry['original'])
                    
                    st.subheader("Optimized Prompt")
                    st.success(entry['optimized'])
                    
                    if st.button(f"Reuse This Prompt", key=f"reuse_{entry['id']}"):
                        st.session_state.current_prompt = entry['original']
                        st.session_state.step = 'input'
                        st.rerun()

def export_history_csv():
    """Export history to CSV for admin"""
    if not st.session_state.history:
        st.warning("No history to export")
        return
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write headers
    writer.writerow(['Timestamp', 'Original Prompt', 'Optimized Prompt', 'Questions', 'Answers'])
    
    # Write data
    for entry in st.session_state.history:
        questions_str = '; '.join(entry.get('questions', []))
        answers_str = '; '.join([f"{q}: {a}" for q, a in entry.get('answers', {}).items()])
        writer.writerow([
            entry['timestamp'],
            entry['original'],
            entry['optimized'],
            questions_str,
            answers_str
        ])
    
    # Create download button
    csv_data = output.getvalue()
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name=f"prompt_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def main():
    """Main application flow"""
    init_session_state()
    
    # Load model if not already loaded
    if st.session_state.model is None:
        st.session_state.model = load_model()
        if st.session_state.model is None:
            st.error("Failed to load model. Please check your configuration.")
            return
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    st.markdown("<h1 class='header-title'>🎯 Prompt Optimization Tool</h1>", unsafe_allow_html=True)
    st.markdown("Transform your prompts into precision-engineered instructions for AI assistants")
    
    # Step 1: Input
    if st.session_state.step == 'input':
        st.header("Step 1: Enter Your Prompt")
        
        # Use existing prompt if available (from history reuse)
        initial_value = st.session_state.current_prompt if st.session_state.current_prompt else ""
        
        prompt = st.text_area(
            "Enter the prompt you want to optimize:",
            value=initial_value,
            height=150,
            max_chars=2000,
            placeholder="Example: Write a blog post about AI"
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Optimize This Prompt", type="primary", use_container_width=True):
                if prompt.strip():
                    if len(prompt) < 10:
                        st.warning("Please enter a more detailed prompt (at least 10 characters)")
                    else:
                        st.session_state.current_prompt = prompt
                        st.session_state.step = 'clarify'
                        st.rerun()
                else:
                    st.warning("Please enter a prompt to optimize")
    
    # Step 2: Clarification
    elif st.session_state.step == 'clarify':
        st.header("Step 2: Clarification Questions")
        
        # Display original prompt
        with st.container():
            st.subheader("Your Original Prompt:")
            st.info(st.session_state.current_prompt)
        
        # Generate questions if not already done
        if st.session_state.clarifying_questions is None:
            st.session_state.clarifying_questions = generate_clarifying_questions(
                st.session_state.current_prompt,
                st.session_state.model
            )
        
        st.subheader("Please answer these clarifying questions to help optimize your prompt:")
        st.caption("Leave blank any questions that don't apply")
        
        # Display questions and collect answers
        with st.form("clarification_form"):
            for i, question in enumerate(st.session_state.clarifying_questions):
                answer = st.text_input(
                    f"{i+1}. {question}",
                    key=f"q_{i}",
                    placeholder="Your answer here (optional)"
                )
                st.session_state.user_answers[question] = answer
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("↩️ Back to Edit Prompt", use_container_width=True):
                    st.session_state.step = 'input'
                    st.rerun()
            with col2:
                if st.form_submit_button("✨ Generate Optimized Prompt", type="primary", use_container_width=True):
                    st.session_state.step = 'optimize'
                    st.rerun()
    
    # Step 3: Optimization
    elif st.session_state.step == 'optimize':
        st.header("Step 3: Your Optimized Prompt")
        
        # Generate optimized prompt if not already done
        if st.session_state.optimized_prompt is None:
            st.session_state.optimized_prompt = optimize_prompt(
                st.session_state.current_prompt,
                st.session_state.user_answers,
                st.session_state.model
            )
            
            # Save to history
            save_to_history(
                st.session_state.current_prompt,
                st.session_state.optimized_prompt,
                st.session_state.clarifying_questions,
                st.session_state.user_answers
            )
        
        # Display optimized prompt
        st.markdown("<div class='optimized-prompt'>", unsafe_allow_html=True)
        st.markdown("### ✨ Optimized Prompt")
        st.write(st.session_state.optimized_prompt)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📋 Copy to Clipboard", type="primary", use_container_width=True):
                st.code(st.session_state.optimized_prompt)
                st.success("✅ Prompt ready to copy!")
        
        with col2:
            if st.button("🔄 Start New Prompt", use_container_width=True):
                reset_workflow()
                st.rerun()
        
        with col3:
            if st.button("📊 View History", use_container_width=True):
                st.session_state.sidebar_state = 'expanded'
                st.rerun()
        
        # Show comparison
        with st.expander("📊 See the Transformation"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.text(st.session_state.current_prompt)
            with col2:
                st.subheader("Optimized")
                st.text(st.session_state.optimized_prompt)

if __name__ == "__main__":
    main()
