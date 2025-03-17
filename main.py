import discord
from discord.ext import commands
from fireworks.client import Fireworks
import os
from dotenv import load_dotenv
import logging
from pymongo import MongoClient
from nomic import embed
import numpy as np
from setup_mongodb import setup_mongodb

# Set up logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
collection = setup_mongodb()

# Create bot instance with command prefix '!' and all intents
intents = discord.Intents.all()  # Enable all intents
bot = commands.Bot(command_prefix='!', intents=intents)

def split_response(response, max_length=1900):
    """Split a long response into chunks that fit within Discord's message limit"""
    chunks = []
    current_chunk = ""
    
    # Split by sentences to keep context
    sentences = response.split('. ')
    
    for sentence in sentences:
        # Add the period back except for the last sentence if it doesn't have one
        if sentence != sentences[-1] or response.endswith('.'):
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
    
    return chunks

async def get_relevant_context(question, max_results=3):
    """Get relevant context from MongoDB using vector similarity search"""
    try:
        # Generate embedding for the question using Nomic
        logger.info("Generating embedding for question...")
        output = embed.text(
            texts=[question],
            model='nomic-embed-text-v1.5',
            task_type='search_query',
            dimensionality=384  # Match MongoDB index dimensions
        )
        question_embedding = output['embeddings'][0]
        
        # Perform vector similarity search
        logger.info("Performing vector similarity search...")
        pipeline = [
            {
                "$search": {
                    "index": "vector_index",
                    "knnBeta": {
                        "vector": question_embedding,
                        "path": "embedding",
                        "k": max_results
                    }
                }
            },
            {
                "$project": {
                    "plot": 1,
                    "title": 1,
                    "fullplot": 1,
                    "score": {"$meta": "searchScore"}
                }
            }
        ]
        
        results = list(collection.aggregate(pipeline))
        
        if results:
            context = "\n\nRelevant movie information:\n"
            for result in results:
                # Include title and plot information
                context += f"\nTitle: {result.get('title', 'N/A')}\n"
                context += f"Plot: {result.get('plot', result.get('fullplot', 'No plot available'))}\n"
                context += f"Relevance Score: {result.get('score', 'N/A')}\n"
                context += "-" * 50 + "\n"
            return context
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
    """Ask a question about movies"""
    logger.info(f"Received question from {ctx.author}: {question}")
    try:
        # Show typing indicator while processing
        async with ctx.typing():
            # Get relevant context from MongoDB
            context = await get_relevant_context(question)
            
            # Create chat completion with Fireworks
            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful AI assistant specializing in movies. 
                    When provided with movie information, use it to enhance your answers while maintaining a natural conversational tone.
                    If the movie context is not relevant to the question, rely on your general knowledge about movies. Don't answer questions completely unrelated to this."""
                }
            ]
            
            # Add context if available
            if context:
                messages.append({
                    "role": "system",
                    "content": context
                })
            
            messages.append({
                "role": "user",
                "content": question
            })
            
            response = fireworks_client.chat.completions.create(
                model="accounts/fireworks/models/llama-v3p3-70b-instruct",
                messages=messages,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            logger.info(f"Generated response: {ai_response[:100]}...")
            
            if not ai_response:
                ai_response = "I apologize, but I couldn't generate a proper response. Please try asking your question differently."

            # Split long responses into chunks
            response_chunks = split_response(ai_response)
            
            # Send the first chunk as a reply
            await ctx.reply(response_chunks[0])
            
            # Send any remaining chunks as follow-up messages
            for chunk in response_chunks[1:]:
                await ctx.send(chunk)

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        await ctx.reply("Sorry, I encountered an error while processing your question. Please try again.")

@bot.command(name='ping')
async def ping(ctx):
    """Simple command to check if bot is responsive"""
    logger.info(f"Ping command received from {ctx.author}")
    await ctx.send(f'Pong! Latency: {round(bot.latency * 1000)}ms')

if __name__ == "__main__":
    logger.info("Starting bot...")
    bot.run(TOKEN) 