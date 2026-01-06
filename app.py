"""
RetNet LLM Streamlit Application
A comprehensive interface for text generation and model training.
updated

"""

import streamlit as st
import torch
import os
import time
import plotly.graph_objects as go
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader

from retnet_model import RetNet, RetNetConfig
from model_utils import load_checkpoint, initialize_model, get_model_info, get_device, get_device_info, save_checkpoint
from training_engine import LocalTextDataset, TrainingEngine


# Page configuration
st.set_page_config(
    page_title="RetNet LLM Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Note: Max upload size (3000 MB) is configured in .streamlit/config.toml

# Minimal CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'config' not in st.session_state:
        st.session_state.config = None
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None
    if 'checkpoint_info' not in st.session_state:
        st.session_state.checkpoint_info = None
    if 'device' not in st.session_state:
        st.session_state.device = get_device()
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    if 'training_active' not in st.session_state:
        st.session_state.training_active = False
    if 'training_metrics' not in st.session_state:
        st.session_state.training_metrics = []


init_session_state()


# Sidebar
with st.sidebar:
    st.header("RetNet Studio")
    
    st.subheader("Configuration")
    
    # Device selection
    device_info = get_device_info()
    if device_info['cuda_available']:
        device_option = st.selectbox(
            "Device",
            ["cuda", "cpu"],
            index=0,
            help=f"GPU: {device_info.get('cuda_device_name', 'Unknown')}"
        )
    else:
        device_option = "cpu"
        st.info("CUDA not available. Using CPU.")
    
    st.session_state.device = device_option
    
    st.markdown("---")
    
    # Model loading section
    st.subheader("Model Management")
    
    # Checkpoint source selection
    checkpoint_source = st.radio(
        "Checkpoint Source",
        ["Pre-loaded Checkpoints", "Upload from System"],
        help="Choose whether to load a pre-existing checkpoint or upload a new one"
    )
    
    checkpoint_path = None
    
    if checkpoint_source == "Pre-loaded Checkpoints":
        # Get list of checkpoints from checkpoints folder
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if checkpoint_files:
                selected_checkpoint = st.selectbox(
                    "Select Checkpoint",
                    checkpoint_files,
                    help="Choose from available pre-loaded checkpoints"
                )
                checkpoint_path = os.path.join(checkpoint_dir, selected_checkpoint)
            else:
                st.warning("No checkpoint files found in the checkpoints folder.")
        else:
            st.warning("Checkpoints folder not found.")
    else:
        # Checkpoint uploader
        checkpoint_file = st.file_uploader(
            "Upload Checkpoint (.pt)",
            type=['pt'],
            help="Upload a trained RetNet checkpoint file"
        )
        
        if checkpoint_file is not None:
            # Save uploaded file temporarily
            temp_path = f"temp_checkpoint_{int(time.time())}.pt"
            with open(temp_path, 'wb') as f:
                f.write(checkpoint_file.read())
            checkpoint_path = temp_path
    
    # Load checkpoint button (works for both sources)
    if checkpoint_path is not None:
        if st.button("Load Checkpoint", use_container_width=True):
            with st.spinner("Loading checkpoint..."):
                try:
                    # Load checkpoint
                    model, config, checkpoint_info = load_checkpoint(checkpoint_path, st.session_state.device)
                    
                    # Initialize tokenizer
                    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                    tokenizer.pad_token = tokenizer.eos_token
                    
                    # Update session state
                    st.session_state.model = model
                    st.session_state.config = config
                    st.session_state.tokenizer = tokenizer
                    st.session_state.checkpoint_info = checkpoint_info
                    
                    # Clean up temp file if it was uploaded
                    if checkpoint_source == "Upload from System" and os.path.exists(checkpoint_path):
                        os.remove(checkpoint_path)
                    
                    st.success(f"Checkpoint loaded! Step: {checkpoint_info['global_step']}")
                except Exception as e:
                    st.error(f"Error loading checkpoint: {str(e)}")
                    # Clean up temp file on error
                    if checkpoint_source == "Upload from System" and checkpoint_path.startswith("temp_checkpoint_"):
                        if os.path.exists(checkpoint_path):
                            os.remove(checkpoint_path)
    
    # Initialize new model
    if st.button("Initialize New Model", use_container_width=True):
        with st.spinner("Initializing model..."):
            try:
                config = RetNetConfig()
                model = initialize_model(config, st.session_state.device)
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                tokenizer.pad_token = tokenizer.eos_token
                
                st.session_state.model = model
                st.session_state.config = config
                st.session_state.tokenizer = tokenizer
                st.session_state.checkpoint_info = None
                
                st.success("New model initialized!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    
    # Model status
    if st.session_state.model is not None:
        st.subheader("Model Status")
        st.success("Model Loaded")
        if st.session_state.checkpoint_info:
            st.metric("Training Step", st.session_state.checkpoint_info['global_step'])
            if st.session_state.checkpoint_info['best_val_loss'] != float('inf'):
                st.metric("Val Loss", f"{st.session_state.checkpoint_info['best_val_loss']:.4f}")
    else:
        st.subheader("Model Status")
        st.warning("No model loaded")


# Main content
st.title("RetNet Language Model Studio")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Chat & Generation", "Training", "Model Info"])


# Tab 1: Chat & Generation
with tab1:
    st.header("Text Generation")
    
    if st.session_state.model is None:
        st.warning("Please load a checkpoint or initialize a new model from the sidebar.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Input")
            prompt = st.text_area(
                "Enter your prompt:",
                height=150,
                placeholder="The future of artificial intelligence...",
                help="Enter the text prompt to generate from"
            )
        
        with col2:
            st.subheader("Generation Parameters")
            
            max_tokens = st.slider(
                "Max Tokens",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Maximum number of tokens to generate"
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                value=0.8,
                step=0.1,
                help="Higher = more creative, Lower = more focused"
            )
            
            top_k = st.slider(
                "Top-k",
                min_value=0,
                max_value=100,
                value=50,
                step=5,
                help="Consider top-k most likely tokens (0 = disabled)"
            )
            
            top_p = st.slider(
                "Top-p (Nucleus)",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.05,
                help="Cumulative probability threshold"
            )
            
            repetition_penalty = st.slider(
                "Repetition Penalty",
                min_value=1.0,
                max_value=2.0,
                value=1.2,
                step=0.1,
                help="Penalize repeated tokens"
            )
        
        if st.button("Generate", use_container_width=True, type="primary"):
            if not prompt.strip():
                st.error("Please enter a prompt!")
            else:
                with st.spinner("Generating..."):
                    try:
                        start_time = time.time()
                        
                        # Tokenize input
                        input_ids = st.session_state.tokenizer.encode(
                            prompt,
                            return_tensors='pt'
                        ).to(st.session_state.device)
                        
                        # Generate
                        output_ids = st.session_state.model.generate(
                            input_ids,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            top_k=top_k if top_k > 0 else None,
                            top_p=top_p,
                            repetition_penalty=repetition_penalty
                        )
                        
                        # Decode output
                        generated_text = st.session_state.tokenizer.decode(
                            output_ids[0],
                            skip_special_tokens=True
                        )
                        
                        # Clean up text
                        generated_text = generated_text.replace('<unk>', '').strip()
                        
                        elapsed_time = time.time() - start_time
                        
                        # Display result
                        st.subheader("Generated Text")
                        st.success(generated_text)
                        
                        st.info(f"Generated in {elapsed_time:.2f}s | Tokens: {len(output_ids[0])}")
                        
                        # Add to history
                        st.session_state.generation_history.append({
                            'prompt': prompt,
                            'output': generated_text,
                            'params': {
                                'temperature': temperature,
                                'max_tokens': max_tokens,
                                'top_k': top_k,
                                'top_p': top_p
                            }
                        })
                        
                    except Exception as e:
                        st.error(f"Generation error: {str(e)}")
        
        # Generation history
        if st.session_state.generation_history:
            st.markdown("---")
            st.subheader("Recent Generations")
            
            for i, item in enumerate(reversed(st.session_state.generation_history[-5:])):
                with st.expander(f"Generation {len(st.session_state.generation_history) - i}"):
                    st.markdown(f"**Prompt:** {item['prompt']}")
                    st.markdown(f"**Output:** {item['output']}")
                    st.caption(f"Temp: {item['params']['temperature']} | Tokens: {item['params']['max_tokens']}")


# Tab 2: Training
with tab2:
    st.header("Model Training")
    
    if st.session_state.model is None:
        st.warning("Please load a checkpoint or initialize a new model from the sidebar.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Training Data")
            
            uploaded_file = st.file_uploader(
                "Upload Training Text File",
                type=['txt'],
                help="Upload a .txt file with training data"
            )
            
            if uploaded_file is not None:
                text_content = uploaded_file.read().decode('utf-8')
                st.success(f"Loaded {len(text_content)} characters")
                st.text_area("Preview:", text_content[:500] + "...", height=150, disabled=True)
        
        with col2:
            st.subheader("Training Parameters")
            
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=1e-5,
                max_value=1e-2,
                value=3e-4,
                format="%.6f",
                help="Initial learning rate"
            )
            
            batch_size = st.selectbox(
                "Batch Size",
                [1, 2, 4, 8],
                index=2,
                help="Training batch size"
            )
            
            max_steps = st.number_input(
                "Max Steps",
                min_value=10,
                max_value=10000,
                value=1000,
                step=100,
                help="Maximum training steps"
            )
            
            save_interval = st.number_input(
                "Save Interval",
                min_value=50,
                max_value=1000,
                value=200,
                step=50,
                help="Steps between checkpoint saves"
            )
            
            checkpoint_dir = st.text_input(
                "Checkpoint Directory",
                value="./checkpoints",
                help="Directory to save checkpoints"
            )
        
        # Training controls
        col_start, col_stop = st.columns(2)
        
        with col_start:
            start_training = st.button(
                "Start Training",
                use_container_width=True,
                type="primary",
                disabled=uploaded_file is None or st.session_state.training_active
            )
        
        with col_stop:
            stop_training = st.button(
                "Stop Training",
                use_container_width=True,
                disabled=not st.session_state.training_active
            )
        
        # Training execution
        if start_training and uploaded_file is not None:
            st.session_state.training_active = True
            st.session_state.training_metrics = []
            
            try:
                # Create dataset
                dataset = LocalTextDataset(
                    text_content,
                    seq_len=st.session_state.config.max_seq_len,
                    tokenizer=st.session_state.tokenizer
                )
                
                train_loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True
                )
                
                # Initialize training engine
                engine = TrainingEngine(
                    st.session_state.model,
                    st.session_state.config,
                    device=st.session_state.device,
                    learning_rate=learning_rate
                )
                
                # Training progress containers
                progress_bar = st.progress(0)
                metrics_container = st.empty()
                chart_container = st.empty()
                
                # Training loop
                for metrics in engine.train_step(
                    train_loader,
                    max_steps=max_steps,
                    save_interval=save_interval,
                    checkpoint_dir=checkpoint_dir
                ):
                    if stop_training or not st.session_state.training_active:
                        st.session_state.training_active = False
                        break
                    
                    st.session_state.training_metrics.append(metrics)
                    
                    # Update progress
                    progress_bar.progress(metrics['progress'])
                    
                    # Display metrics
                    with metrics_container.container():
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Step", f"{metrics['step']}/{max_steps}")
                        col2.metric("Loss", f"{metrics['loss']:.4f}")
                        col3.metric("Learning Rate", f"{metrics['lr']:.2e}")
                        col4.metric("Grad Norm", f"{metrics['grad_norm']:.3f}")
                    
                    # Update chart
                    if len(st.session_state.training_metrics) > 1:
                        steps = [m['step'] for m in st.session_state.training_metrics]
                        losses = [m['loss'] for m in st.session_state.training_metrics]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=steps,
                            y=losses,
                            mode='lines',
                            name='Training Loss',
                            line=dict(color='#667eea', width=2)
                        ))
                        fig.update_layout(
                            title="Training Loss",
                            xaxis_title="Step",
                            yaxis_title="Loss",
                            height=300,
                            margin=dict(l=0, r=0, t=30, b=0)
                        )
                        chart_container.plotly_chart(fig, use_container_width=True)
                
                st.session_state.training_active = False
                st.success("Training completed!")
                
            except Exception as e:
                st.session_state.training_active = False
                st.error(f"Training error: {str(e)}")


# Tab 3: Model Info
with tab3:
    st.header("Model Information")
    
    if st.session_state.model is None:
        st.warning("Please load a checkpoint or initialize a new model from the sidebar.")
    else:
        model_info = get_model_info(st.session_state.model, st.session_state.config)
        
        # Model statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Parameters", f"{model_info['total_parameters_millions']:.2f}M")
        
        with col2:
            st.metric("Model Size", f"{model_info['model_size_mb']:.2f} MB")
        
        with col3:
            st.metric("Device", st.session_state.device.upper())
        
        st.markdown("---")
        
        # Architecture details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Architecture")
            arch = model_info['architecture']
            st.json({
                "Vocabulary Size": arch['vocab_size'],
                "Model Dimension": arch['d_model'],
                "Number of Layers": arch['n_layers'],
                "Number of Heads": arch['n_heads'],
                "FFN Dimension": arch['ffn_dim'],
                "Max Sequence Length": arch['max_seq_len']
            })
        
        with col2:
            st.subheader("Checkpoint Info")
            if st.session_state.checkpoint_info:
                st.json({
                    "Training Step": st.session_state.checkpoint_info['global_step'],
                    "Best Val Loss": f"{st.session_state.checkpoint_info['best_val_loss']:.4f}",
                    "File Size": f"{st.session_state.checkpoint_info['file_size_mb']:.2f} MB"
                })
            else:
                st.info("No checkpoint loaded (new model)")
        
        # Device info
        st.markdown("---")
        st.subheader("Device Information")
        device_info = get_device_info()
        st.json(device_info)


# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; padding: 1rem;">'
    'RetNet LLM Studio | Built with Streamlit & PyTorch'
    '</div>',
    unsafe_allow_html=True
)
