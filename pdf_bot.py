import discord
from discord.ext import commands
from fireworks.client import Fireworks
import os
from dotenv import load_dotenv
import logging
from pymongo import MongoClient
from nomic import embed
import numpy as np
from pdf_setup_mongodb import setup_mongodb_pdf

# Set up more detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pdf_bot.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get the tokens and check if they exist
TOKEN = os.getenv('DISCORD_TOKEN')
NOMIC_API_KEY = os.getenv('NOMIC_API_KEY')

if not TOKEN:
    logger.error("No Discord token found in .env file!")
    exit(1)
if not NOMIC_API_KEY:
    logger.error("No Nomic API key found in .env file!")
    exit(1)

# Initialize Fireworks client
fireworks_client = Fireworks(api_key=os.getenv('FIREWORKS_API_KEY'))

# Initialize MongoDB
collection = setup_mongodb_pdf()

# Create bot instance with command prefix '!' and all intents
intents = discord.Intents.all()
bot = commands.Bot(command_prefix='!', intents=intents)

def split_response(response, max_length=1900):
    """Split a long response into chunks that fit within Discord's message limit"""
    chunks = []
    current_chunk = ""
    
    # Clean up any double periods that might exist in the original response
    response = response.replace("..", ".")
    
    # Split by sentences to keep context
    sentences = response.split('. ')
    
    for i, sentence in enumerate(sentences):
        # Add the period back except for the last sentence if it doesn't have one
        if i < len(sentences) - 1 or response.endswith('.'):
            # Check if sentence already ends with a period
            if not sentence.endswith('.'):
                sentence += '.'
            
        # If adding this sentence would exceed the limit, start a new chunk
        if len(current_chunk) + len(sentence) + 1 > max_length:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ' '
        else:
            current_chunk += sentence + ' '
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Final cleanup to ensure no double periods
    chunks = [chunk.replace("..", ".") for chunk in chunks]
    
    logger.info(f"Split response into {len(chunks)} chunks")
    return chunks

async def get_relevant_context(question, max_results=5):
    """Get relevant context from MongoDB using vector similarity search"""
    try:
        # Generate embedding for the question
        output = embed.text(
            texts=[question],
            model='nomic-embed-text-v1.5',
            task_type='search_query',
            dimensionality=384 
        )
        question_embedding = output['embeddings'][0]
        
        # Perform vector similarity search
        pipeline = [
            {
                "$search": {
                    "index": "discord_bot_ai",
                    "knnBeta": {
                        "vector": question_embedding,
                        "path": "embedding",
                        "k": max_results
                    }
                }
            },
            {
                "$project": {
                    "content": 1,
                    "metadata": 1,
                    "score": {"$meta": "searchScore"}
                }
            }
        ]
        
        results = list(collection.aggregate(pipeline))
        logger.info(f"Found {len(results)} relevant documents")
        
        if results:
            context = "\n\nRelevant documentation sections:\n"
            for i, result in enumerate(results, 1):
                file_name = result['metadata']['file_name']
                chunk_num = result['metadata'].get('chunk_number', 'N/A')
                total_chunks = result['metadata'].get('total_chunks', 'N/A')
                score = result.get('score', 'N/A')                
                logger.info(f"Result {i}: File={file_name}, Score={score:.2f}")
                context += f"\nFrom file: {file_name} (Chunk {chunk_num}/{total_chunks})\n"
                context += f"Content: {result['content']}\n"
                context += f"Relevance Score: {score}\n"
                context += "-" * 50 + "\n"
            return context
        
        logger.warning("No relevant documents found in vector search")
        return ""
        
    except Exception as e:
        logger.error(f"Error getting context: {str(e)}")
        return ""

@bot.event
async def on_ready():
    logger.info(f'Bot is ready! Logged in as {bot.user.name} (ID: {bot.user.id})')
    logger.info(f'Connected to {len(bot.guilds)} guilds:')
    for guild in bot.guilds:
        logger.info(f'- {guild.name} (ID: {guild.id})')

@bot.event
async def on_connect():
    logger.info("Bot connected to Discord!")

@bot.event
async def on_disconnect():
    logger.warning("Bot disconnected from Discord!")

@bot.event
async def on_error(event, *args, **kwargs):
    logger.error(f'Error in {event}: {args} {kwargs}')

@bot.command(name='ask')
async def ask(ctx, *, question):
    """Ask a question about Fireworks AI models"""
    logger.info(f"Question from {ctx.author}: {question[:50]}...")
    try:
        # Show typing indicator while processing
        async with ctx.typing():
            # Get relevant context from MongoDB
            context = await get_relevant_context(question)
            
            # Create chat completion with Fireworks
            messages = [
                {
                    "role": "system",
                    "content": """You are a customer support engineer for Fireworks AI models. Provide direct, concise, and precise answers without fluff.

                    Context:
                    - All questions are about models offered by Fireworks AI
                    - Assume users are asking about Fireworks-specific implementations
                    - Focus on Fireworks' specific features, capabilities, and parameters
                    
                    Response Guidelines:
                    - For simple questions (what/is/does): Keep responses under 100 words, just the facts
                    - For "how to" questions: Provide step-by-step instructions with necessary detail
                    - For "why" questions: Explain reasoning but stay focused on the core answer
                    - For "compare" questions: Create structured comparisons with key differences
                    
                    Adapt your response length to the complexity of the question - be brief for simple questions and more detailed for complex ones.

                    Do not:
                    - Reference documentation chunks, files, or implementation details
                    - Include references, citations, or links
                    - Use phrases like "I can help you with" or "feel free to ask"
                    - Mention your access to documentation

                    Do:
                    - Give factual, technically accurate information about Fireworks AI models
                    - Be clear and direct
                    - Use a professional tone
                    - Answer exactly what was asked
                    - Structure complex answers with headings or bullet points for readability"""
                }
            ]
            
            # Add context if available
            if context:
                messages.append({
                    "role": "system",
                    "content": context
                })
            
            # Analyze question complexity to guide response length
            is_complex_question = any(marker in question.lower() 
                                     for marker in ["why", "how", "explain", "compare", 
                                                   "difference", "versus", "pros and cons"])
                
            if is_complex_question:
                # Add guidance for a more detailed response
                question_with_guidance = f"{question}\n\nThis requires a detailed explanation."
            else:
                # Add guidance for a concise response
                question_with_guidance = f"{question}\n\nKeep the response brief and to the point."
            
            messages.append({
                "role": "user",
                "content": question_with_guidance
            })
            
            # Get response from Fireworks
            response = fireworks_client.chat.completions.create(
                model="accounts/fireworks/models/llama-v3p3-70b-instruct",
                messages=messages,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            if not ai_response:
                logger.warning("Empty response from Fireworks API")
                ai_response = "I apologize, but I couldn't generate a proper response. Please try asking your question differently."

            # Split long responses into chunks
            response_chunks = split_response(ai_response)
            
            # Send the first chunk as a reply
            await ctx.reply(response_chunks[0])
            
            # Send any remaining chunks as follow-up messages
            for chunk in response_chunks[1:]:
                await ctx.send(chunk)

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        await ctx.reply("Sorry, I encountered an error while processing your question. Please try again.")

@bot.command(name='ping')
async def ping(ctx):
    """Simple command to check if bot is responsive"""
    logger.info(f"Ping command received from {ctx.author}")
    await ctx.send(f'Pong! Latency: {round(bot.latency * 1000)}ms')

if __name__ == "__main__":
    logger.info("Starting bot...")
    bot.run(TOKEN) 