# Local AI Email Agent

A completely local-first, conversational AI email assistant built with **LangGraph**, **Ollama**, and the **Gmail API**. 

This agent runs directly from your terminal and allows you to manage your Gmail inbox using natural language. Because it uses a local LLM (Ollama), **none of your email contents are sent to third-party AI providers** like OpenAI or Anthropic — ensuring complete privacy.

---
##  The Problem Statement & My Contributions

**The Modern Inbox Dilemma:** Standard email clients rely entirely on rigid, exact-keyword matching (lexical search) and require manual triage to stay organized. If you are looking for emails about "learning," a standard client will miss an email from Udemy titled "Masterclass on Python" because the exact keyword isn't present. Furthermore, actionable items and deadlines are frequently buried inside long threads, forcing context-switching between your inbox and your to-do lists.

**The Solution (New Integrations):** I forked this project to transform it from a simple conversational email reader into an **autonomous, semantically-aware inbox manager**. I implemented the following major system upgrades:

1. **Semantic Search Engine (RAG):** Replaced strict keyword reliance with a local ChromaDB vector database. The agent now mathematically embeds your emails, allowing you to search by *abstract concept or meaning*.
2. **Google Tasks Automation:** Engineered an extraction pipeline that reads complex threads, identifies deadlines/action items, and autonomously syncs them to your Google Tasks board.
3. **Automated Daily Triage:** Developed a standalone CLI batch job (`auto_triage.py`) that acts as a background assistant—fetching emails, applying smart labels, executing task extraction, and sending a beautifully formatted HTML digest of its actions directly to your inbox.

---

## Features

- **Search & List:** Ask the agent to find specific emails (e.g., "Find my unread emails from Google"). The agent translates this into valid Gmail search queries under the hood.
- **Read & Summarize:** Read the full body of any email directly in your terminal. You can also ask the agent to summarize specific emails or your top 5 unread messages.
- **Label Management:** Organize your inbox naturally. Archive emails, mark them as read/unread, or star/unstar important messages without having to delete them (e.g., "Archive email 1", "Star the 3rd email").
- **Attachment Handling:** 
  - **Read & Summarize:** The agent automatically parses text from attached PDF, CSV, or TXT files so it can include them in its summaries and context.
  - **Download:** Ask the agent to download an attachment, and it will be saved locally to a `downloads` directory.
  - **Send with Attachments:** Ask the agent to attach local files when composing an email.
- **Send Emails:** Compose and send emails using natural language. The agent will ask for any missing fields (To, Subject, Body) and will always ask for your confirmation before dispatching.
- **Delete/Trash:** Keep your inbox clean by asking the agent to delete specific emails or bulk-delete spam based on your descriptions (e.g., "Delete all emails from newsletters@spam.com").
- **Beautiful CLI:** Built with `Typer` and `Rich` for styled panels, dynamic execution status updates, and a clean reading experience.
- **Google Tasks Sync:** Ask the agent to "Extract tasks from this thread," and it will populate your connected Google Tasks list.
- **Daily HTML Digests:** The `auto_triage.py` script runs silently, organizes specific categories of emails, and emails you a clean HTML summary log.
---

## Architecture

- **Orchestration:** `LangGraph` (state machine for conversational routing and memory).
- **LLM Engine:** `Ollama` running `llama3.1` locally.
- **API Integration:** Official Google `gmail-api`, `tasks-api`.
- **Vector Database (RAG):** `ChromaDB` (Local) & `sentence-transformers` (`all-MiniLM-L6-v2`).
- **Interface:** `Typer` + `Rich`.

---

## Setup & Installation

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running on your machine.
- A Google Cloud Platform (GCP) project with the **Gmail API** and **Google Tasks API** enabled.

### 2. Pull the Local Model
Start Ollama and pull the required model:
```bash
ollama run llama3.1
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
- `gmail.modify` (to move emails to the trash and apply labels)

Once authorized, a `token.pickle` file will be generated locally.

### Example Prompts
Here are a few things you can type in the interactive prompt:

- "List my unread emails."
- "Search for emails from github this week."
- "Read email number 2."
- "Summarize those emails."
- "Mark the first email as read and archive it."
- "Download the attachment from email 2."
- "Send an email to ansh@example.com about the meeting tomorrow and attach C:\path\to\report.pdf"
- "Delete the 3rd email."

Type `exit` or `quit` to leave the agent.

### The Semantic Vector Sync (scripts/vector_sync.py)

When to use it: Before you attempt to run any Semantic Searches (RAG). You should run this script periodically (e.g., once a week or every morning) to ensure your local vector database is up to date with your latest emails.

What happens: Standard search engines look for exact words. This script bypasses that limitation by downloading your recent emails and running them through a local sentence-transformers machine learning model. It converts the text into mathematical vectors and stores them safely in your local email_vector_db/ directory. This allows the AI to understand the meaning of your inbox.

How to run:

```bash
python scripts/vector_sync.py
```

Note: Depending on the volume of your inbox, the initial sync may take a few minutes as the local model processes the text.

### Automated Daily Triage (auto_triage.py)

When to use it: For automated, background inbox management. This script is designed to be run as a cron job or scheduled task. It fetches emails matching your criteria, applies labels, extracts actionable tasks, and emails you a clean, compiled HTML digest of its actions.

The digest email (rendered from `templates/digest.html`) includes:
- **Agent actions** — a log of what the triage performed (labels applied, tasks added, etc.)
- **Email cards** — for each matched email: the **subject**, **sender**, a one-sentence **AI summary**, and a direct **"Open in Gmail"** link (`https://mail.google.com/mail/u/0/#all/{message_id}`) to jump straight to the source message

What happens: The script runs a single-pass LangGraph execution. It does not require user interaction. It processes the emails based on the flags you provide and immediately terminates upon completion.

Command Structure:

```bash
python auto_triage.py "CATEGORY_OR_KEYWORDS" [FLAGS]
```

### Flag Configuration

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `CATEGORY` | String | Required | The target subject (e.g., "internships", "newsletters", "urgent"). |
| `--label` | String | None | Automatically applies a specified Gmail label (e.g., STARRED, IMPORTANT). |
| `--tasks` | Boolean | False | Triggers the LLM to extract deadlines/action items and sync to Google Tasks. |
| `--time` | String | 1d | Historical timeframe (e.g., 1d, 7d, 1m). |
| `--max` | Integer | 10 | The maximum number of emails to process in this batch. |
| `--semantic` | Boolean | False | Enables vector database (ChromaDB) queries for conceptual matches. |

### Example Workflows

**Workflow A: The Lexical Triage (Standard)**
Perfect for catching exact phrases or specific senders over the last week.
```bash
python auto_triage.py "internship, application, offer" --label STARRED --tasks --time 7d --max 20
```

Result: Fetches up to 20 emails from the last 7 days containing those exact words, stars them, adds extracted deadlines to Google Tasks, and emails you a summary.

**Workflow B: The Semantic Triage (Advanced RAG)**
(Requires running vector_sync.py first)
```bash
python scripts/vector_sync.py
```

Perfect for finding abstract concepts where exact keywords might be missing.

```bash
python auto_triage.py "career progression and networking" --semantic --max 10
```

Result: Queries your local vector database for 10 emails related to the concept of career progression, intelligently summarizes the abstract findings, and sends you a daily digest.

---

### File-Based Triage (auto_triage_file.py)

An alternative to `auto_triage.py` designed for cron jobs and scheduled tasks. Reads keywords from a `.txt` file and automatically runs `vector_sync` before every triage, keeping the semantic database current without manual intervention.

#### File Format
Create a plain text file with one keyword per line. Lines starting with `#` are ignored:
```bash
# internships.txt
internship
application
offer
remote
```

#### Usage (Cron-Ready)
```bash
python auto_triage_file.py --file internships.txt --label STARRED --tasks --time 7d --max 20
```

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--file, -f` | String | **Required** | Path to the `.txt` keywords file |
| `--label, -l` | String | None | Gmail label to apply (e.g., STARRED, IMPORTANT) |
| `--tasks, -t` | Bool | False | Extract deadlines into Google Tasks |
| `--time` | String | 1d | Search timeframe (1d, 7d, 1m) |
| `--max` | Integer | 20 | Max emails to process |
| `--semantic, -s` | Bool | False | Use vector DB for conceptual search |
| `--skip-sync` | Bool | False | Skip the automatic vector DB sync at startup |

The vector database is synced automatically on every run (unless `--skip-sync` is set), so you don't need to schedule `vector_sync.py` separately.

---

### Windows Task Scheduler Automation (run_triage.ps1)

A PowerShell wrapper script (`run_triage.ps1`) is included for Windows Task Scheduler. It handles the full pipeline — starting Ollama if needed, waiting for it to be ready, then running the triage.

#### Script Behavior
1. Checks if `ollama.exe` is running; if not, starts `ollama serve` in the background
2. Polls `http://localhost:11434/api/tags` up to 30 seconds until Ollama responds
3. Exits with code 1 if Ollama fails to start (triggers Task Scheduler retry)
4. Runs `auto_triage_file.py` with your keyword file

#### Setup Steps

1. **Create your keywords file** — one keyword per line, `#` for comments:
   ```bash
   keywords.txt
   ```
2. **Open Task Scheduler** (<kbd>Win</kbd>+<kbd>R</kbd> → `taskschd.msc`)
3. **Create Task** with these settings:

   | Tab | Setting | Value |
   |:---|:---|:---|
   | **General** | Name | `Email Agent Daily Triage` |
   | | Run whether user is logged on or not | ✅ |
   | | Run with highest privileges | ✅ |
   | **Triggers** | Daily at | `7:00 AM` (or your preferred time) |
   | **Actions** | Program/script | `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe` |
   | | Arguments | `-ExecutionPolicy Bypass -File "C:\path\to\email-agent\run_triage.ps1"` |
   | | Start in | `C:\path\to\email-agent` |
   | **Conditions** | Stop if on battery | ❌ (uncheck for laptops) |
   | **Settings** | Restart on failure | Every 1 minute, up to 3 times |

4. **Test** — right-click the task → **Run**, check the **History** tab for results.

> **Note:** If Ollama is already set to start on login (default), the script detects it and skips the startup step.

---

### Project Structure
```bash
EMAIL-AGENT/
├── agent/
│   ├── graph.py             # Main LangGraph state machine & logic
│   └── __init__.py
├── email_vector_db/         # Local ChromaDB storage (Ignored in Git)
├── gmail/
│   ├── auth.py              # OAuth2 authentication
│   ├── client.py            # API client initialization
│   └── tools.py             # Search, read, send, and label tools
├── scripts/
│   ├── vector_sync.py       # Script to embed emails into ChromaDB
│   └── test_semantic.py     # Sandbox script to test vector retrieval
├── tasks/
│   ├── tools.py             # Google Tasks API wrappers
├── templates/
│   └── digest.html          # Jinja2 HTML template for the daily triage digest email
├── utils/
│   └── parser.py            # Attachment parsing (PDF, CSV, TXT)
├── vector/
│   ├── tools.py             # ChromaDB initialization and search queries
├── auto_triage.py           # CLI script for automated batch jobs (terminal keywords)
├── auto_triage_file.py      # CLI script for automated batch jobs (keywords from .txt file)
├── run_triage.ps1           # PowerShell wrapper for Windows Task Scheduler (starts Ollama + runs triage)
├── main.py                  # Entry point for the interactive chat CLI
├── requirements.txt
└── .gitignore
```

## Privacy Notice
Your `credentials.json` and `token.pickle` grant direct access to your Gmail account. **Never commit these files to version control.** They are included in the `.gitignore` by default. All LLM reasoning is done entirely locally via Ollama.
