FSM-Based AI Chatbot

📌 Overview

This project implements a Finite State Machine (FSM)-based AI-powered chatbot designed for automated customer support. The bot efficiently manages user queries by transitioning between predefined states, utilizing an LLM (Large Language Model) for intelligent responses, and escalating issues when necessary. The system also includes a logging mechanism and a knowledge base for adaptive learning.

🚀 **(Note: This is just a prototype and not a fully developed customer support chatbot.  
With future improvements, it can be adapted for real-world applications.)**(Note: Just a prototype)

🚀 Features

FSM-driven Workflow: Handles conversations through structured state transitions.

LLM-Powered Responses: Uses an AI model to generate human-like responses.

Mathematical Query Handling: Directly solves arithmetic operations.

Escalation Mechanism: Detects user frustration and escalates to a human agent if required.

Conversation Logging: Records interactions for analytics.

Knowledge Base Integration: Stores learned intents and resolution patterns to improve future responses.

📂 Project Structure

├── fsm.py    # Main chatbot implementation
├── llm_knowledge_base.json # Knowledge base for storing interaction data (Cuurently Not Working Will Fix This Issue)
├── .env                  # Environment variables (API keys, etc.)
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation

🛠️ Technologies Used

Python (Core language)

Transitions (FSM implementation)

Groq API (LLM-based responses)

Logging (For debugging and analytics)

JSON (Knowledge base storage)

🔧 Installation

Clone the repository:

git clone https://github.com/your-repo/fsm-ai-chatbot.git
cd fsm-ai-chatbot

Set up a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install dependencies:

pip install -r requirements.txt

Set up API keys:

Create a .env file in the project root and add:

GROQ_API_KEY=your_api_key_here

▶️ Usage

Run the chatbot using:

python llm_support_bot.py

The chatbot will start an interactive session where you can enter queries.

🎯 How It Works

User Interaction Begins → Chatbot enters INITIAL_GREETING state.

Context Understanding → Determines the user's intent.

LLM Analysis → Generates an appropriate response using the LLM.

Solution Generation & Confirmation → Provides an answer and checks if the user is satisfied.

Escalation (if needed) → Transfers control to a human agent if user frustration is detected.

Closure → Ends the conversation.

🏗️ Future Improvements

Implement a multi-turn dialogue memory for context retention.

Add intent classification for faster response mapping.

Enhance adaptive learning to refine responses over time.

Integrate with external APIs for fetching dynamic information.

🤝 Contributing

If you'd like to contribute:

Fork the repository.

Create a feature branch.

Submit a pull request.

📞 Contact

For any queries, feel free to reach out at your-yashbisen24@gmail.com.
