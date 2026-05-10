# MediBot - Medical AI Assistant

A Streamlit-based RAG (Retrieval-Augmented Generation) chatbot that provides medical information from local PDF documents using FAISS vector search and Hugging Face models.

## Features

- **Local Medical Knowledge Base**: Answers questions based on medical encyclopedia content
- **RAG Architecture**: Combines retrieval from vector database with generative AI
- **Streamlit Web Interface**: Easy-to-use chat interface
- **Source Citations**: Shows relevant source documents for transparency
- **CPU-Only Operation**: Runs on CPU without requiring GPU

## Setup Instructions

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your Hugging Face API token:
   ```
   HF_TOKEN=your_actual_token_here
   ```

4. **Add medical PDFs**:
   Place your medical PDF documents in the `data/` directory.

5. **Build the vector store**:
   ```bash
   python create_memory_for_llm.py
   ```

6. **Run the application**:
   ```bash
   # Windows
   .\run_medibot.bat

   # Or manually
   python -m streamlit run medibot.py --server.port 8502
   ```

7. **Access the app**:
   Open your browser to `http://localhost:8502`

## Project Structure

```
├── medibot.py              # Main Streamlit application
├── create_memory_for_llm.py # Script to build FAISS vector store
├── connect_memory_with_llm.py # Alternative query script
├── vector_utils.py         # Utility functions for vector operations
├── run_medibot.bat         # Windows launch script
├── requirements.txt        # Python dependencies
├── Pipfile                 # Alternative dependency management
├── .env.example           # Environment variables template
├── data/                  # Directory for PDF documents
├── vectorstore/           # Generated FAISS index (auto-created)
└── .gitignore            # Git ignore rules
```

## Usage

1. Start the application using the run script
2. Ask medical questions in the chat interface
3. The bot will provide answers based on the medical documents
4. View source citations by expanding the "View Sources" section

## Configuration

### Environment Variables
- `HF_TOKEN`: Hugging Face API token for model access

### Model Settings
- **Model**: google/flan-t5-base (T5 architecture)
- **Max Response Length**: 2048 tokens
- **Temperature**: 0.1 (deterministic responses)

### Vector Store
- **Chunk Size**: 500 characters
- **Chunk Overlap**: 50 characters
- **Embedding Model**: sentence-transformers (default)

## Development

### Adding New Documents
1. Place PDF files in `data/` directory
2. Run `python create_memory_for_llm.py` to rebuild the vector store
3. Restart the Streamlit app

### Modifying the Prompt
Edit the `CUSTOM_PROMPT_TEMPLATE` in `medibot.py` to change how the AI responds.

## Security Notes

- Never commit `.env` files containing real API tokens
- The `.gitignore` file excludes sensitive files
- Use `.env.example` as a template for required environment variables

## Troubleshooting

### Common Issues

1. **"Failed to load LLM"**: Check your HF_TOKEN in `.env`
2. **Empty responses**: Ensure PDFs are in `data/` and vector store is built
3. **Import errors**: Run `pip install -r requirements.txt`

### Performance
- First run may be slow due to model loading
- Vector store building takes time for large documents
- Responses are generated on CPU (may be slower than GPU)

## License

This project is for educational and research purposes. Medical information should not replace professional medical advice.