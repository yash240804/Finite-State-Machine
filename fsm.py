import os
import json
import logging
import re
from typing import Dict, Any, List
from datetime import datetime
import transitions
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_support_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LLMSupportBot:
    states = [
        'INITIAL_GREETING',
        'CONTEXT_UNDERSTANDING',
        'LLM_ANALYSIS',
        'SOLUTION_GENERATION',
        'INTENT_VERIFICATION',
        'DEEP_REASONING',
        'HUMAN_ESCALATION',
        'RESOLUTION_CONFIRMATION',
        'CLOSURE'
    ]

    def __init__(self, knowledge_base_path='llm_knowledge_base.json'):
        try:
            self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        except Exception as e:
            logger.error(f"Groq client initialization error: {e}")
            raise

        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = self.load_knowledge_base()
        
        self.interaction_context = {
            'conversation_history': [],
            'current_intent': None,
            'escalation_triggers': 0
        }
        
        self.machine = transitions.Machine(
            model=self, 
            states=self.states, 
            initial='INITIAL_GREETING',
            auto_transitions=False
        )
        
        self.setup_state_transitions()

    def setup_state_transitions(self):
        """Configure advanced state transitions"""
 
        self.machine.add_transition(
            trigger='start_interaction', 
            source='INITIAL_GREETING', 
            dest='CONTEXT_UNDERSTANDING',
            conditions=['validate_initial_input']
        )
        
        self.machine.add_transition(
            trigger='analyze_context', 
            source='CONTEXT_UNDERSTANDING', 
            dest='LLM_ANALYSIS'
        )

        self.machine.add_transition(
            trigger='generate_solution', 
            source='LLM_ANALYSIS', 
            dest='SOLUTION_GENERATION'
        )

        self.machine.add_transition(
            trigger='escalate', 
            source=['CONTEXT_UNDERSTANDING', 'LLM_ANALYSIS', 'SOLUTION_GENERATION'], 
            dest='HUMAN_ESCALATION',
            conditions=['detect_escalation_need']
        )
   
        self.machine.add_transition(
            trigger='confirm_resolution', 
            source='SOLUTION_GENERATION', 
            dest='RESOLUTION_CONFIRMATION'
        )

        self.machine.add_transition(
            trigger='close', 
            source=['RESOLUTION_CONFIRMATION', 'HUMAN_ESCALATION'], 
            dest='CLOSURE'
        )

    def validate_initial_input(self, user_input: str) -> bool:
        """Initial input validation"""
        if not user_input or len(user_input.strip()) < 3:
            return False

        try:
            if re.search(r'\d+\s*[+\-*/]\s*\d+', user_input):
                return True
            
            return len(user_input.strip()) >= 3
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return False

    def detect_escalation_need(self, user_input: str) -> bool:
        """Simple escalation detection"""
        frustration_keywords = [
            'angry', 'frustrated', 'upset', 'not working', 
            'terrible service', 'need help immediately'
        ]
        
        return any(kw in user_input.lower() for kw in frustration_keywords)

    def call_llm(self, prompt: str, model: str = "llama-3.1-8b-instant") -> str:
        """Centralized LLM call method with improved error handling"""
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful customer support assistant."},
                    {"role": "user", "content": prompt}
                ],
                model=model,
                max_tokens=300,
                temperature=0.7
            )
            
            return chat_completion.choices[0].message.content
        
        except Exception as e:
            logger.error(f"LLM call error: {e}")
            return "I'm having trouble processing your request. Could you please rephrase?"

    def generate_solution(self, user_input: str) -> str:
        """Solution generation with fallback for simple queries"""
        # Handle mathematical queries directly
        math_match = re.match(r'(\d+)\s*([+\-*/])\s*(\d+)', user_input)
        if math_match:
            try:
                num1 = int(math_match.group(1))
                num2 = int(math_match.group(3))
                op = math_match.group(2)
                
                if op == '+':
                    return f"The answer is: {num1 + num2}"
                elif op == '-':
                    return f"The answer is: {num1 - num2}"
                elif op == '*':
                    return f"The answer is: {num1 * num2}"
                elif op == '/':
                    return f"The answer is: {num1 / num2}"
            except Exception as e:
                logger.error(f"Math calculation error: {e}")
                return "I couldn't perform the calculation."
        
        # Fallback to LLM for complex queries
        solution_prompt = f"""
        Generate a comprehensive solution for the following user input:
        User Query: {user_input}
        
        Provide a clear, concise response that directly addresses the query.
        """
        
        return self.call_llm(solution_prompt)

    def load_knowledge_base(self) -> Dict[str, Any]:
        """Load or initialize knowledge base"""
        try:
            if os.path.exists(self.knowledge_base_path):
                with open(self.knowledge_base_path, 'r') as f:
                    return json.load(f)
            else:
                base_structure = {
                    'interaction_logs': [],
                    'learned_intents': {},
                    'resolution_patterns': {}
                }
                with open(self.knowledge_base_path, 'w') as f:
                    json.dump(base_structure, f, indent=4)
                return base_structure
        except Exception as e:
            logger.error(f"Knowledge base loading error: {e}")
            return {'interaction_logs': [], 'learned_intents': {}, 'resolution_patterns': {}}

    def save_knowledge_base(self):
        """Save the updated knowledge base to the JSON file."""
        try:
            with open(self.knowledge_base_path, 'w') as f:
                json.dump(self.knowledge_base, f, indent=4)
            logger.info("Knowledge base successfully saved.")
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")

    def run_interaction_workflow(self):
        """Main LLM-powered interaction workflow"""
        print("🤖 Welcome to Advanced LLM-Powered Support!")
        
        while self.state != 'CLOSURE':
            try:
                user_input = input("You: ").strip()
 
                if self.state == 'INITIAL_GREETING':
                    if self.validate_initial_input(user_input):
                        print("🤖 Hello! How can I assist you today?")
                        self.start_interaction(user_input)
                        self.analyze_context(user_input)
                
                elif self.state == 'CONTEXT_UNDERSTANDING':
                    if self.detect_escalation_need(user_input):
                        self.escalate(user_input)
                    else:
                        self.analyze_context(user_input)
                
                elif self.state == 'LLM_ANALYSIS':
                    solution = self.generate_solution(user_input)
                    logger.info(f"Chatbot Response: {solution}")
                    print(f"🤖 {solution}")
                    self.generate_solution(user_input)
                
                elif self.state == 'SOLUTION_GENERATION':
                    print("🤖 Let me verify if this solution meets your needs.")
                    self.confirm_resolution(user_input)
                
                elif self.state == 'RESOLUTION_CONFIRMATION':
                    if any(word in user_input.lower() for word in ['yes', 'good', 'great', 'perfect']):
                        print("🤖 Great! I'm glad I could help.")
                        self.close()
                    else:
                        print("🤖 I apologize. Let me try a different approach.")
                        self.start_interaction(user_input)
                
                elif self.state == 'HUMAN_ESCALATION':
                    print("👥 A human agent will assist you shortly.")
                    self.close()
                
                elif self.state == 'CLOSURE':
                    print("👋 Thank you for using our support service!")
                    break
            
            except KeyboardInterrupt:
                print("\n🤖 Interaction terminated.")
                break

def main():
    """Main execution function"""
    support_bot = LLMSupportBot()
    support_bot.run_interaction_workflow()

if __name__ == "__main__":
    main()