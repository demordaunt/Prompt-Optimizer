"""
Lightweight version optimized for free tier hosting
Uses smaller models and simplified processing
"""

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
from datetime import datetime
import time
from typing import List, Dict

# Page configuration
st.set_page_config(
    page_title="Prompt Optimizer - Lite",
    page_icon="🎯",
    layout="centered"
)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'step' not in st.session_state:
    st.session_state.step = 1

@st.cache_resource
def load_model():
    """Load a lightweight model for free tier"""
    try:
        # Using Flan-T5 base - much lighter than Mistral/Llama
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        model_name = "google/flan-t5-base"  # 250M parameters, runs on CPU
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_questions(prompt: str, tokenizer, model) -> List[str]:
    """Generate clarifying questions using lightweight approach"""
    
    input_text = f"""Generate 5 clarifying questions for this prompt: "{prompt}"
    
    Questions should ask about:
    1. The main goal
    2. Target audience
    3. Format needed
    4. Any constraints
    5. Level of detail
    """
    
    try:
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse response or use defaults
        questions = [
            "What is the primary goal or objective?",
            "Who is the target audience?",
            "What format should the output be in?",
            "Are there any specific constraints or requirements?",
            "What level of detail is needed?"
        ]
        
        return questions
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        # Return default questions
        return [
            "What is the primary goal?",
            "Who is the target audience?",
            "What format do you need?",
            "Any specific requirements?",
            "How detailed should it be?"
        ]

def optimize_prompt(original: str, answers: Dict[str, str], tokenizer, model) -> str:
    """Create optimized prompt using template approach"""
    
    # Build context from answers
    context_parts = []
    for q, a in answers.items():
        if a and a.strip():
            context_parts.append(f"• {q}: {a}")
    
    context = "\n".join(context_parts)
    
    input_text = f"""Rewrite this prompt to be clear and specific:
    
Original: {original}

Additional context:
{context}

Optimized prompt:"""
    
    try:
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=300,
                num_return_sequences=1,
                temperature=0.3
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # If model response is too short or unclear, use template-based approach
        if len(response) < 20:
            optimized = f"{original}\n\nContext and Requirements:\n{context}"
        else:
            optimized = response
        
        return optimized
        
    except Exception as e:
        # Fallback to template-based optimization
        return f"{original}\n\nContext and Requirements:\n{context}"

def main():
    st.title("🎯 Prompt Optimizer (Lite Version)")
    st.caption("Free tier optimized - Transform your prompts into clear instructions")
    
    # Load model
    tokenizer, model = load_model()
    
    if tokenizer is None or model is None:
        st.error("Failed to load model. Please refresh the page.")
        return
    
    # Sidebar with history
    with st.sidebar:
        st.header("📝 Recent Prompts")
        
        if st.session_state.history:
            for i, entry in enumerate(st.session_state.history[-5:]):  # Show last 5
                with st.expander(f"{entry['time']} - {entry['original'][:30]}..."):
                    st.text("Original:")
                    st.write(entry['original'])
                    st.text("Optimized:")
                    st.success(entry['optimized'])
        else:
            st.info("No history yet")
    
    # Main workflow
    if st.session_state.step == 1:
        st.header("Step 1: Enter Your Prompt")
        
        prompt = st.text_area(
            "What prompt would you like to optimize?",
            height=100,
            placeholder="Example: Write a blog post about AI"
        )
        
        if st.button("🚀 Optimize", type="primary"):
            if prompt and len(prompt) > 10:
                st.session_state.prompt = prompt
                st.session_state.questions = generate_questions(prompt, tokenizer, model)
                st.session_state.step = 2
                st.rerun()
            else:
                st.warning("Please enter a longer prompt")
    
    elif st.session_state.step == 2:
        st.header("Step 2: Quick Clarification")
        st.info(f"Original: {st.session_state.prompt}")
        
        with st.form("questions_form"):
            answers = {}
            for i, q in enumerate(st.session_state.questions):
                answers[q] = st.text_input(
                    f"{i+1}. {q}",
                    key=f"q{i}",
                    placeholder="Optional - leave blank if not applicable"
                )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("← Back"):
                    st.session_state.step = 1
                    st.rerun()
            with col2:
                if st.form_submit_button("Generate ✨", type="primary"):
                    st.session_state.answers = answers
                    st.session_state.step = 3
                    st.rerun()
    
    else:  # Step 3
        st.header("✨ Your Optimized Prompt")
        
        # Generate optimized version
        if 'optimized' not in st.session_state:
            with st.spinner("Optimizing..."):
                st.session_state.optimized = optimize_prompt(
                    st.session_state.prompt,
                    st.session_state.answers,
                    tokenizer,
                    model
                )
                
                # Save to history
                st.session_state.history.append({
                    'time': datetime.now().strftime("%H:%M"),
                    'original': st.session_state.prompt,
                    'optimized': st.session_state.optimized
                })
        
        # Display result
        st.success("Here's your optimized prompt:")
        st.code(st.session_state.optimized, language=None)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Copy Above Text"):
                st.info("Select and copy the text above")
        with col2:
            if st.button("🔄 Start New"):
                for key in ['prompt', 'questions', 'answers', 'optimized']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.step = 1
                st.rerun()

if __name__ == "__main__":
    main()
