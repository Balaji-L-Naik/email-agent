# ✉ Local AI Email Agent

A completely local-first, conversational AI email assistant built with **LangGraph**, **Ollama**, and the **Gmail API**. 

This agent runs directly from your terminal and allows you to manage your Gmail inbox using natural language. Because it uses a local LLM (Ollama), **none of your email contents are sent to third-party AI providers** like OpenAI or Anthropic — ensuring complete privacy.

---

## Features

- **Search & List:** Ask the agent to find specific emails (e.g., *"Find my unread emails from Google"*). The agent translates this into valid Gmail search queries under the hood.
- **Read:** Read the full body of any email directly in your terminal.
- **Summarize:** Too long, didn't read? Ask the agent to summarize specific emails or your top 5 unread messages.
- **Send:** Compose and send emails using natural language. The agent will ask for any missing fields (To, Subject, Body) and will **always ask for your confirmation** before dispatching.
- **Delete/Trash:** Keep your inbox clean by asking the agent to delete specific emails or bulk-delete spam based on your descriptions (e.g., *"Delete all emails from newsletters@spam.com"*).
- **Beautiful CLI:** Built with `Typer` and `Rich` for styled panels, interactive spinners, and a clean reading experience.

---

## 🛠️ Architecture

- **Orchestration:** `LangGraph` (state machine for conversational routing and memory).
- **LLM Engine:** `Ollama` running `qwen2.5-coder:3b` locally.
- **API Integration:** Official Google `gmail-api`.
- **Interface:** `Typer` + `Rich`.

---

## Setup & Installation

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running on your machine.
- A Google Cloud Platform (GCP) project with the **Gmail API** enabled.

### 2. Pull the Local Model
Start Ollama and pull the required model:
```bash
ollama run qwen2.5-coder:3b
```

### 3. Clone & Install
```bash
git clone <your-repo-url>
cd email-agent

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Setup Gmail Credentials
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project and enable the **Gmail API**.
3. Go to **APIs & Services > Credentials** and create an **OAuth client ID** (Desktop Application).
4. Download the JSON file, rename it to `credentials.json`, and place it in the root of this project folder.
   > *Note: `credentials.json` is safely ignored by `.gitignore` so you don't accidentally push it to GitHub.*

---

## Usage

Start the agent by running:

```bash
python main.py
```

On your **first run**, a browser window will open asking you to authorize the app. It requires the following scopes:
- `gmail.readonly` (to read and search emails)
- `gmail.send` / `gmail.compose` (to send emails)
- `gmail.modify` (to move emails to the trash)

Once authorized, a `token.pickle` file will be generated locally.

### Example Prompts
Here are a few things you can type in the interactive prompt:

- *"List my top 5 unread emails."*
- *"Search for emails from github this week."*
- *"Read email number 2."*
- *"Summarize those emails."*
- *"Send an email to john@example.com about the meeting tomorrow."*
- *"Delete the 3rd email."*
- *"Delete all my emails from spammy-marketing@example.com"*

Type `exit` or `quit` to leave the agent.

---

## 🔒 Privacy Notice
Your `credentials.json` and `token.pickle` grant direct access to your Gmail account. **Never commit these files to version control.** They are included in the `.gitignore` by default. All LLM reasoning is done entirely locally via Ollama.
