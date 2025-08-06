Swafinix AI Hackathon 2025: AI Admission Inquiry Assistant
This project is an AI-powered agent designed to answer admission inquiries for educational institutions. It leverages Retrieval-Augmented Generation (RAG) to provide accurate, context-aware answers by processing unstructured policy documents. The system is built for the Swafinix AI Hackathon 2025 and demonstrates a robust solution for automating information retrieval in real-world scenarios.

Features
RAG-based Question Answering: Uses a combination of a Language Model (Mistral AI) and a vector database (FAISS) to retrieve relevant information from policy documents and generate accurate answers.

Dynamic Document Processing: Automatically loads, chunks, and indexes PDF documents on application startup, eliminating the need for manual pre-processing.

Simple API Endpoint: Exposes a single, clean API endpoint (/inquire) to handle natural language queries.

Automation Integration: Integrates with n8n via a webhook to trigger automated workflows, such as sending email notifications for every query.

Efficient and Scalable: The in-memory FAISS vector store provides fast similarity search, making the system performant even with large document sets.

Clear and Concise Output: The API returns a simple JSON object containing only the generated answer, designed for easy integration with downstream applications.

Project Structure
swafinix-admission-ai-assistant/
├── data/
│   └── admission_policies/         # Place your PDF documents here
│       └── admission_policy_2025.pdf
├── main.py                         # All application logic is here
├── requirements.txt
├── .env.example
├── README.md
├── Dockerfile
├── render.yaml
└── .gitignore
How to Run Locally
Follow these steps to set up and run the application on your local machine.

1. Clone the Repository
Bash

git clone [https://github.com/your-username/swafinix-admission-ai-assistant.git](https://github.com/Anmolj0815/adm-ai.git)
cd swafinix-admission-ai-assistant
2. Set Up Environment Variables
Create a .env file in the project's root directory based on the .env.example file and fill in your API keys.

MISTRAL_API_KEY=your_mistral_ai_api_key
# The N8N_WEBHOOK_URL is optional for local testing but required for the n8n integration.
N8N_WEBHOOK_URL=your_n8n_webhook_url_here
3. Install Dependencies
It's recommended to use a virtual environment.

Bash

pip install -r requirements.txt
4. Place Your Documents
Add all your admission policy PDFs to the data/admission_policies directory. The application will index these documents automatically on startup.

5. Run the Application
Bash

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
The API will be available at http://localhost:8000.

API Endpoint
The application exposes a single API endpoint to process queries.

Endpoint: /inquire

Method: POST

Request Body:

JSON

{
    "query": "What is the minimum bachelor's degree percentage required for a candidate in the SC category?"
}
Response:

JSON

{
    "answer": "A candidate in the Scheduled Caste (SC) category must have at least 45% marks or equivalent CGPA in their bachelor's degree."
}
Deployment
This project is configured for easy deployment on Render.com using the provided Dockerfile and render.yaml.

Push your code to a GitHub repository.

Go to your Render dashboard and create a new Web Service.

Connect your repository and choose to deploy from the main branch.

Render will automatically use the Dockerfile and render.yaml to build and deploy your application.

Set your environment variables (MISTRAL_API_KEY, N8N_WEBHOOK_URL) in the Render dashboard under Environment.

The startup script will automatically process your PDFs and prepare the RAG system for incoming requests.

N8N Integration
This project is set up to trigger an n8n workflow for every query.

In your n8n account, create a new workflow with a Webhook node.

Get the Production URL from the webhook node.

Set this URL as the N8N_WEBHOOK_URL environment variable in your .env file and on the Render dashboard.

Configure your workflow's "Send email" node with expressions like {{ $json.original_query }} and {{ $json.answer }} to receive the data from your API.
