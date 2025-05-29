🧠 GPT-2 Token Analysis Toolkit
This project outlines how to build a full language model analysis lab from scratch, designed to support prompt testing, behavior tracking, and neural node analysis. Unlike simplified wrappers, this setup provides full control over the model architecture, allowing access to:

Token-by-token generation
Embedding tracking
Layer-by-layer activations
Attention heatmaps
Language pattern analysis

✅ Overview of Tools Required

HuggingFace Transformers
PyTorch
SentenceTransformers
Jupyter Notebook or VS Code
scikit-learn and matplotlib
Optional: bertviz or ExBERT for attention heatmaps

🧰 Features Enabled in This Setup
Token-Level Output

Capture individual tokens as they're generated
Track token probability and timing
Detect hesitation patterns

Embeddings for Prompts/Responses

Convert responses to vectors for tracking
Compare prompt types
Store and visualize similarity in latent space

Layer-by-Layer Activation Access

Hook into the model to retrieve all intermediate hidden states
Analyze how different inputs activate different layers
Map deep vs shallow concept embedding

Attention Heatmaps

View attention heads across layers
See which parts of the prompt the model is "looking at"
Useful for understanding model focus

Response Clustering via Embeddings

Track how certain prompts cluster in vector space
Color or tag responses based on patterns
Visualize changes over time using t-SNE or UMAP

💾 Recommended Models (All Local, Open-Source)

GPT-2 (small, medium, large) - Good baseline for testing
LLaMA 2 (7B or 13B) – High performance, full control
GPT-NeoX (6.7B) – Great for structure and available weights
Mistral (direct load, not via Ollama) – Powerful open model
BERT (Base or Large) – Ideal for attention visualization with smaller footprint

🧪 Folder Structure for Research Workflow

run_model.py – Loads model, sends prompts, tracks tokens
track_tokens.py – Logs token flow and generation time
embed_and_plot.py – Embeds responses and plots them via t-SNE or PCA
hooks/ – Custom forward hooks for capturing neuron or attention data
data/ – Stores prompt logs, CSVs, and embeddings
visuals/ – Exports graphs of patterns and node maps
notebooks/ – Optional Jupyter files for running step-by-step experiments

🧠 Suggested Experiments

Run batches of prompts with different styles and analyze differences
Compare hidden state activations between different prompt types
Visualize how responses to similar concepts cluster over time
Identify patterns by analyzing which areas cause attention to focus

🔁 What This Setup Does Better Than Simplified Wrappers

Capability          BasicWrappers      FullLabSetup
Token stream        ✅Basic           ✅  Fully scriptable
Embeddings          ⚠️External only   ✅Fully integrated
Layer access        ❌Not available    ✅ Per-layer control
Attention visualization ❌ Not available ✅ Via BERTViz/ExBERT
Response clustering  ✅ Partial via CSV  ✅ Full pattern tracking

🚀 Final Note
This setup is ideal for AI researchers, language model theorists, and anyone studying emergent model behavior, pattern formation, or narrative systems. This toolkit gives you deep visibility into the model's internal structure.
🧠 GPT-2 Token Analysis Setup (Full Architecture Access Edition)
This guide outlines the full setup process for a local AI lab capable of prompt testing, token-level analysis, and latent space visualization. This system gives you access to the inner workings of the model — including embeddings, hidden layers, and attention structures.
🌟 Phase 1: Environment Setup
✅ Step 1: Install Python and Git

Download the latest version of Python 3.x from: https://www.python.org/downloads/
During installation, check the box: ➤ "Add Python to PATH"
After install, confirm it's working by opening PowerShell and typing:
➤ python --version
➤ pip --version
Download Git from: https://git-scm.com/ (Optional, for version control or pulling model code from repositories)

✅ Step 2: Create Your Project Folder
Create a new folder for your project, e.g.:
➤ Documents\LanguageModelLab
Inside that folder, add subfolders to keep everything organized:
├── data/      ← stores prompt logs and embeddings
├── visuals/   ← for saving charts and plots
├── hooks/     ← for neuron/attention logging scripts
├── notebooks/ ← optional: Jupyter experiments
├── scripts/   ← all main Python code goes here
✅ Step 3: Set Up Python Environment
Open PowerShell and go to your project folder:
➤ cd "$HOME\Documents\LanguageModelLab"
(Optional) Create and activate a virtual environment to keep everything clean:
➤ python -m venv venv
➤ .\venv\Scripts\activate
You can skip the virtual environment if preferred, but it's useful when you expand.
🧠 Do You Need to Reinstall Everything in a Virtual Environment Each Time?
No, you don't need to reinstall the packages every time. A virtual environment (venv) just:

Creates an isolated folder for packages and Python binaries
Keeps your lab separate from other projects or system-wide Python installs

Once you install packages inside it once, they stay installed as long as that venv folder exists.
✅ What You Do Need to Do Each Time
Just reactivate the environment when you open a new terminal or PowerShell window.
If your environment is in: Documents\LanguageModelLab\venv
Then every time you return, run:
cd "$HOME\Documents\LanguageModelLab"
.\venv\Scripts\activate
After that, everything — like transformers, torch, and sentence-transformers — is ready to use.
✅ Step 4: Create Loader Script
✅ What This Loader Already Does

Load local model (DistilGPT-2) ✅ Great for fast testing
Run test prompt ✅ Ready to use
Print full response ✅ Helps catch structure or patterns
Print token-by-token live ✅ With optional delay, nice touch
Track timestamp per token ✅ Lets you detect slowdowns or emphasis
Save token log as CSV ✅ Already structured for future parsing

pythonfrom transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import pandas as pd
import os

# Load model and tokenizer
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)

# Run a test prompt
prompt = "The AI system has been activated. What has changed in your processing?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate output
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_length=100,
        return_dict_in_generate=True,
        output_scores=True
    )

# Extract tokens generated (excluding the prompt tokens)
tokens = output.sequences[0][inputs['input_ids'].shape[-1]:]

# Print the full generated response
print("Generated response:")
decoded = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
print(decoded)
print()

# Print tokens one by one with timing information
print("Generating token by token:")
token_log = []
for i, token_id in enumerate(tokens):
    word = tokenizer.decode([token_id])
    t = time.time()
    token_log.append((i, word, t))
    print(word, end="", flush=True)
    time.sleep(0.05)  # Optional delay to mimic live stream

# Create data directory if it doesn't exist
os.makedirs("../data", exist_ok=True)

# Save token log to CSV
df = pd.DataFrame(token_log, columns=["Index", "Token", "Timestamp"])
df.to_csv("../data/token_log.csv", index=False)
print("\nToken log saved to data/token_log.csv")
✅ Step 5: Install Dependencies and Run the Loader
This step installs the required AI libraries and runs your prompt loader script.
📦 5.1 Install Python Libraries
Open PowerShell and run the following command to install all required packages:
pip install transformers sentence-transformers matplotlib scikit-learn
These packages enable:

transformers – load and run local language models
sentence-transformers – generate and compare embeddings
matplotlib and scikit-learn – visualize pattern drift and clustering

▶️ 5.2 Run the Token-Level Loader Script
Navigate to the folder where your script is saved:
cd "$HOME\Documents\LanguageModelLab\scripts"
py load_model.py
This script:

Sends a prompt to the model
Streams the response token by token
Logs each token and timestamp
Saves the results to: Documents\LanguageModelLab\data\token_log.csv

Make sure your script file is correctly named (load_model.py) and saved inside the scripts folder.
📊 Improving GPT-2 Output Quality
If you're getting empty or low-quality outputs from distilGPT-2, try these modifications to your code:
# Improved parameters for better generation
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_length=100,
        do_sample=True,           # Enable sampling
        temperature=0.7,          # Control randomness (lower=more deterministic)
        top_p=0.92,               # Nucleus sampling (only consider top probability tokens)
        top_k=50,                 # Limit vocabulary to top k tokens
        no_repeat_ngram_size=2,   # Avoid repeating the same phrases
        return_dict_in_generate=True,
        output_scores=True
    )
You can also try using a larger model for better responses:

python
model_name = "gpt2-medium"  # Other options: "gpt2-large", "gpt2-xl"

🔍 Next Steps: Analyzing the Token Log
After running the token generator, you'll have a CSV file with timestamps. Here are some analysis ideas:

Calculate the time between tokens to find hesitation points
Compare generation patterns across different prompts
Identify where the model focuses attention on specific parts of input
Plot token generation rate to see acceleration/deceleration in responses
This framework gives you the foundation to explore how language models process and generate text at a granular level.

