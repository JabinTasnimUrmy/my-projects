# RepoFinder AI: Intelligent GitHub Project Discovery

<div align="center">  
</div>

![alt text](demo.gif)

<p align="center">
  An intelligent, full-stack web application that translates natural language project ideas into technical search queries, and then uses semantic analysis to discover the most relevant existing repositories on GitHub.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10-blue.svg"/>
  <img alt="Flask" src="https://img.shields.io/badge/Flask-Web%20Framework-green.svg"/>
  <img alt="Groq" src="https://img.shields.io/badge/AI%20Model-Llama%203-purple.svg"/>
  <img alt="Sentence Transformers" src="https://img.shields.io/badge/NLP-Semantic%20Search-orange.svg"/>
</p>

---

## 📜 Table of Contents

*   [About The Project](#-about-the-project)
*   [Key Features](#-key-features)
*   [Tech Stack](#-tech-stack)
*   [Project Structure](#-project-structure)
*   [Getting Started](#-getting-started)
*   [System Architecture & AI Pipeline](#-system-architecture--ai-pipeline)
*   [Showcase: The Power of AI in Action](#-showcase-the-power-of-ai-in-action)
*   [Future Scope](#-future-scope)


---

## About The Project

Developers often struggle to find existing projects on GitHub. The search bar requires precise technical keywords, but ideas often start as conversational, high-level concepts. This project bridges that gap.

**RepoFinder AI** acts as an intelligent assistant. Instead of just matching keywords, it employs a sophisticated two-stage AI pipeline to truly understand a user's intent:

1.  **AI Query Translator:** A powerful Large Language Model (Groq Llama 3) analyzes the user's conversational idea and translates it into a set of optimized, technical search queries that an expert developer would use.
2.  **Semantic Similarity Ranker:** After gathering a rich pool of candidate projects, a Sentence Transformer model calculates the semantic "distance" between the user's original, detailed idea and the description of each repository, finding the single best match.

The entire system is wrapped in a clean, responsive web interface built with Flask and modern JavaScript, demonstrating a complete full-stack development cycle.

---

## Key Features

*   **Natural Language Understanding:** Accepts conversational, long-form project ideas.
*   **AI-Powered Query Generation:** Intelligently translates vague concepts into expert-level search terms.
*   **Semantic Ranking:** Goes beyond keywords to find the most contextually relevant project.
*   **Full-Stack Web Interface:** A responsive and user-friendly UI built with Flask and vanilla JavaScript.
*   **Automatic Browser Launch:** The server script automatically opens the application in a new browser tab for an improved developer experience.

---

## Tech Stack

| Category      | Technology                                                                                                    |
| :------------ | :------------------------------------------------------------------------------------------------------------ |
| **Backend**   | Python, Flask, Groq API, GitHub REST API                                                                      |
| **Frontend**  | HTML5, CSS3, JavaScript (ES6)                                                                                 |
| **AI / ML**   | Llama 3 (via Groq for Query Generation), Sentence-Transformers `all-MiniLM-L6-v2` (for Semantic Ranking)      |
| **Dev Tools** | Git & GitHub, Virtualenv, VS Code                                                                             |

---

## 📂 Project Structure

The project is organized with a clear separation between the backend AI logic and the frontend interface, following industry best practices.

```
repofinder-chatbot/
│
├── src/                               # Folder for all core backend Python modules
│   ├── github_client.py               # Module to communicate with the GitHub API
|   |── chatbot_logic.py               # Chatbot that searches GitHub and finds the closest matching project to idea.
│   ├── intelligent_query_generator.py # AI module to translate user ideas to search queries
│   └── semantic_analyzer.py           # AI module to rank search results by meaning
|   
│
├── static/                            # Folder for frontend assets that don't change
│   ├── style.css                      # The CSS stylesheet for the web page
│   └── script.js                      # The JavaScript to handle user interaction and API calls
│
├── templates/                         # Folder for Flask to find HTML files
│   └── index.html                     # The main HTML structure of the web page
│
├── app.py                             # The main application file: runs the Flask web server
├── config.py                          # Loads secret API keys from the .env file
├── requirements.txt                   # Lists all Python libraries needed for the project
├── .env                               # Private file for secret API keys (do not share)
└── README.md                          # Project's documentation
```

---

## Getting Started

Follow these steps to set up and run the project locally.

### Step 1: Clone the Repository
```bash
git clone <your-repo-link>
cd repofinder-chatbot
```
### Step 2: Create and Activate a Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate
```
### Step 3: Install All Required Packages
```bash
pip install -r requirements.txt
```
### Step 4: Configure API Keys

In the root directory, create a file named .env and add your secret keys.
<details>
<summary><strong>🔑 How to get your GitHub API Token</strong></summary>

1. Log in to your GitHub account.
2. Go to Settings (click your profile picture in the top-right).
3. Scroll down and click on Developer settings in the left sidebar.
4. Click on Personal access tokens, then select Tokens (classic).
5. Click "Generate new token" and select "Generate new token (classic)".
6. Give it a Note (e.g., "RepoFinder AI").
7. Set an Expiration date (e.g., 30 or 90 days).
8. Under "Select scopes", check the box for public_repo.
9. Click "Generate token" and copy the key.
</details>
<details>
<summary><strong>🔑 How to get your Groq API Key</strong></summary>

1. Go to the Groq Console.
2. Sign up for a free account (you can use your Google account).
3. Once logged in, click the "+ Create API Key" button.
4. Give the key a Name (e.g., "repofinder-chatbot").
5. Click "Create" and copy the key that appears.
</details>
<br/>

Add your copied keys to the **.env** file like this:
```bash
GITHUB_API_TOKEN="ghp_YourGitHubTokenHere"
GROQ_API_KEY="gsk_YourGroqApiKeyHere"
```
### Step 5: Run the Web Application
```bash
python app.py
```
The application will automatically open in your default web browser at http://127.0.0.1:5000.

## 🧠 The RepoFinder AI Pipeline: How It Works
The project's intelligence comes from a multi-stage process designed to mimic an expert's workflow.

```

  User's Detailed Idea (e.g., "An app that can listen to a song and tell me its name, like Shazam")
             │
             ▼
┌────────────────────────────┐
│ Stage 1: The Translator    │ (Groq Llama 3 Model)
│ Translates idea into       │
│ technical search terms.    │
└────────────┬───────────────┘
             │
             ▼
  ["audio fingerprinting python", "music recognition algorithm", "shazam clone source code"]
             │
             ▼
┌────────────────────────────┐
│ Stage 2: The Search        │ (GitHub API Client)
│ Gathers a wide pool of     │
│ candidate repositories.    │
└────────────┬───────────────┘
             │
             ▼
  [Repo A, Repo B, Repo C, Repo D, ...] (List of 50+ potential projects)
             │
             ▼
┌────────────────────────────┐
│ Stage 3: The Ranker        │ (Sentence Transformer Model)
│ Compares the user's        │
│ *original idea* to each    │
│ repo description to find   │
│ the highest semantic       │
│ similarity score.          │
└────────────┬───────────────┘
             │
             ▼
  The Single Best Match (e.g., A project named "audio-fingerprint-identifying-python")
             │
             ▼
┌────────────────────────────┐
│ Stage 4: The Interface     │ (Flask + JavaScript)
│ Displays the final result  │
│ in a user-friendly UI.     │
└────────────────────────────┘
```
##  Showcase: The Power of RepoFinder AI in Action
The core innovation of this project is its ability to translate vague, conversational ideas into precise, effective search queries. Here are live results from the application.

### Use Case 1: The Automation & Web Scraping Idea
The model is given a practical, real-world automation task. It successfully identifies the key concepts and finds a relevant project.
![alt text](Image%202.png) 

### Use Case 2: The AI & Machine Learning Idea
Here, the model is given a complex, conceptual goal. It correctly translates "like Shazam" into the underlying technical concepts of "audio fingerprinting" and finds a highly relevant repository.
![alt text](Image%201.png)

## Future Scope
* Code-Level Similarity Search: Implement an additional analysis stage that clones the top repositories and uses Abstract Syntax Trees (ASTs) or code embedding models to find similarities in the actual source code.
* Expanded Data Sources: Integrate with other developer platforms like Stack Overflow or dev.to to find not just code, but also relevant articles and discussions.
* User Feedback Loop: Add a "Was this result helpful?" button to collect user feedback, which could be used to fine-tune a custom AI model in the future.
Dockerization & Deployment: Containerize the entire application using Docker for easy, one-command deployment to cloud services like AWS or Render.