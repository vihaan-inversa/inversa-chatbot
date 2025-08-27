# Deployment Guide for Inversa Chatbot

## Prerequisites

1. **Pinecone Account**: You need a Pinecone account with an API key
2. **OpenAI Account**: You need an OpenAI account with an API key
3. **Render Account**: For hosting the application

## Environment Variables

Set these environment variables in your Render dashboard:

- `PINECONE_API_KEY`: Your Pinecone API key
- `OPENAI_API_KEY`: Your OpenAI API key

## Setup Steps

### 1. Create Pinecone Index

Before deploying, you need to create the Pinecone index. Run this locally:

```bash
# Set your environment variables
export PINECONE_API_KEY="your_pinecone_api_key"
export OPENAI_API_KEY="your_openai_api_key"

# Create the index
python create_index.py
```

### 2. Deploy to Render

1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Set the build command: `pip install -r requirements.txt`
4. Set the start command: `uvicorn app:app --host=0.0.0.0 --port=8000`
5. Add your environment variables

### 3. Upload Documents

After deployment, you can upload your documents by running:

```bash
python embed_pdfs.py
```

## Health Check

Once deployed, you can check the health of your application:

```
GET /health
```

This will show you the status of your Pinecone index and API keys.

## Troubleshooting

### Common Issues

1. **Pinecone Index Not Found**: Run `create_index.py` to create the index
2. **API Key Errors**: Check your environment variables in Render
3. **Import Errors**: Make sure all dependencies are in `requirements.txt`

### Logs

Check the Render logs for any error messages. The application now has better error handling and won't crash on startup if the Pinecone index is unavailable.

## API Endpoints

- `POST /query`: Main chat endpoint
- `GET /health`: Health check endpoint

## Notes

- The application now handles missing Pinecone index gracefully
- You can deploy without the index and create it later
- The health endpoint will show you what's configured and what's missing
