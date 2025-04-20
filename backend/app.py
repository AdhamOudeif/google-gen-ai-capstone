import os
from google import genai
from google.api_core import retry
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from typing import Annotated
from typing_extensions import TypedDict, List
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# import re
import pandas as pd
import emoji
from langdetect import detect
from google.genai import types
from data_contraction import contractions_list


# 1. Define retry logic for rate-limiting errors (429, 503)
def is_retriable(e):
    return isinstance(e, genai.errors.APIError) and e.code in {429, 503}


genai.models.Models.generate_content = retry.Retry(predicate=is_retriable)(
    genai.models.Models.generate_content
)

# 2. Load your Google API Key (set it as an environment variable or replace directly)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # <- make sure this env var is set

if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")


# Parses out and returns the video id
def get_video_id(yt_video_url) -> str:
    """Parses out and returns the video id from standard, shortened, or shorts YouTube URLs."""

    # Ensure the URL has a scheme
    if not yt_video_url.startswith(("http://", "https://")):
        yt_video_url = "https://" + yt_video_url

    url = urlparse(yt_video_url)

    # Checks for standard Youtube URLs
    if url.hostname in ["www.youtube.com", "youtube.com"]:
        query_params = parse_qs(url.query)
        video_id = query_params.get("v", [None])[0]
        print(
            "VIDEO ID::: ",
            video_id,
        )

        if video_id:
            return video_id

        # Checks for "Shorts" URLs
        if url.path.startswith("/shorts/"):
            return url.path.split("/")[2]

    # Checks for shortened Youtube URLs (https://youtu.be/abc123)
    if url.hostname in ["youtu.be"]:
        return url.path.lstrip("/")

    raise ValueError(f"Invalid YouTube URL: {yt_video_url}")

filter_prompt = """
You are an AI assistant helping to filter YouTube comments for video idea generation. Your task is to classify each comment as either "relevant" or "irrelevant" based on whether it provides meaningful input for generating video ideas.

### Instructions:
A comment is "relevant" if it:
- Reflects on or interprets the video’s content or message
- Shares a personal story or reaction that’s emotionally or intellectually engaging
- Offers analysis, critique, or a new angle on what was presented
- Raises a thoughtful question or starts a constructive conversation
- Demonstrates deep resonance with a scene, character, or concept
- Provides useful background information or expands on a point in the video
- Connects some aspect of the video content to other information such as related videos, news articles, or podcasts
- Suggests a specific topic or theme for future videos based on the current content
- Offers constructive feedback or suggestions for improvement
- Asks for clarification or elaboration on a specific point in the video

A comment is "irrelevant" if it is:
- Generic praise or vague sentiment (“great video,” “so true,” “very informative”)
- A complaint with no substance (“too many ads,” “this is boring”)
- Spam, scams, promotions, or self-advertising (“investing in this crypto will make you rich”)
- Off-topic or unrelated to the video's themes
- One-liners or jokes that don’t add interpretive value
- Trolling, sarcasm, or passive-aggressive remarks with no clear point
- Vague or speculative accusations without evidence (“I'm sure this is racist somehow”)
- Emoji-only comments or low-effort meme-speak (“💯🔥😎” or “lmao fr”)
- Comments that are purely personal or irrelevant to the video content (“I had pizza for dinner”)

### Example Input:
Comments:
1. "Can you make a video about AI trends?"  
2. "🔥🔥🔥"  
3. "Great video!"  
4. "I need help understanding the capstone project."
5. "Too many ads!"
6. "LLMs are such an interesting topic."

### Example Output:
1. Relevant  
2. Irrelevant  
3. Irrelevant  
4. Relevant
5. Irrelevant
6. Relevant

### Input Comments:
{comments}

### Output:
"""

def filter_comments_with_llm(comments: list) -> list:
    """Filters comments using an LLM to classify them as relevant or irrelevant."""
    print("\nStage 2.1: Filtering Comments with LLM")
    
    # Prepare the input for the LLM
    formatted_comments = "\n".join([f"{i+1}. {comment}" for i, comment in enumerate(comments)])
    input_prompt = filter_prompt.format(comments=formatted_comments)
    
    # Invoke the LLM
    messages = [{"role": "user", "content": input_prompt}]
    response = llm.invoke(messages)
    
    # Parse the LLM response
    classifications = response.content.split("\n")
    relevant_comments = []
    irrelevant_comments = []

    for comment, classification in zip(comments, classifications):
        if "Relevant" in classification:
            relevant_comments.append(comment)
        else:
            irrelevant_comments.append(comment)
    
    # Print relevant and irrelevant comments
    print("\nRelevant Comments:")
    pprint(relevant_comments)
    
    print("\nIrrelevant Comments:")
    pprint(irrelevant_comments)
    
    return relevant_comments


# Preprocessing function
def preprocess_comments(comments: list) -> list:
    """Preprocesses comments to remove duplicates, emojis, non-English comments, etc."""
    print("\nStage 2: Processing Comments")
    print("Raw Comments:")
    pprint(comments)

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(comments, columns=["comment"])

    # Remove duplicate comments
    df = df.drop_duplicates()

    # Remove comments with emojis
    df["comment"] = df["comment"].apply(lambda x: emoji.replace_emoji(x, replace=" "))

    # Remove comments with URLs
    df["comment"] = df["comment"].replace(r'https?://\S+', '', regex=True)

    # Change comments to lowercase
    df["comment"] = df["comment"].str.lower()

    # Expands all the words with contractions with '
    df = df.replace(contractions_list, regex=True)

    # Remove new lines of comments with white spaces
    df = df.replace(r'\n', ' ', regex=True)

    # Remove comments with punctuation and special characters
    df = df.replace(r'[^\w\s]', '', regex=True)

    # Remove non-English comments
    def is_english(text):
        try:
            return detect(text) == "en"
        except:
            return False

    df = df[df["comment"].apply(is_english)]

    # Remove very short comments (e.g., less than 5 characters)
    df = df[df["comment"].str.len() > 5]

    # Filter irrelevant comments using LLM
    processed_comments = df["comment"].tolist()
    filtered_comments = filter_comments_with_llm(processed_comments)

    print("\nProcessed Comments (Final):")
    pprint(filtered_comments)

    return filtered_comments


# Embedding functions
client = genai.Client(api_key=GOOGLE_API_KEY)


def get_embedding(text: str) -> list[float]:
    """Fetches embedding for a given text using Gemini."""
    response = client.models.embed_content(
        model="models/text-embedding-004",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="semantic_similarity",  # important for this use case
        ),
    )
    return response.embeddings[0].values


def embed_all_comments(comments: list) -> pd.DataFrame:
    """Embeds all comments and returns a DataFrame with embeddings."""
    print("\nStage 3: Vectorization / Embeddings")
    print("Embedding Comments...")
    df = pd.DataFrame({"comment": comments})
    df["embedding"] = df["comment"].apply(get_embedding)
    print("Embeddings Complete.")
    print("\nGenerated Embeddings for Comments:")
    for idx, row in df.iterrows():
        print(
            f"Comment: {row['comment'][:50]}... | Embedding: {row['embedding'][:5]}..."
        )  # Print partial embedding for readability
    return df


def get_top_k_similar_comments(query: str, df: pd.DataFrame, k: int = 10) -> list:
    """Finds the top-k most similar comments to the query."""
    print("\nFinding Top-K Similar Comments")
    query_embedding = np.array(get_embedding(query)).reshape(1, -1)

    # Convert comment embeddings to array
    comment_embeddings = np.stack(df["embedding"].to_numpy())

    # Compute cosine similarity
    similarities = cosine_similarity(comment_embeddings, query_embedding).flatten()

    # Add similarity scores to the DataFrame
    df["similarity"] = similarities
    top_comments = df.sort_values("similarity", ascending=False).head(k)
    print(f"\nTop {k} Similar Comments with Similarity Scores:")
    for idx, row in top_comments.iterrows():
        print(f"Comment: {row['comment']} | Similarity: {row['similarity']:.4f}")
    return top_comments["comment"].tolist()


# Fetch top level comments, default max_result=50
def fetch_comments(video_id: str, max_results: int = 50, query: str = None) -> list:
    """Fetches, preprocesses, and finds top-k similar comments."""
    print("\nStage 1: Scraping YouTube Comments")
    yt = build("youtube", "v3", developerKey=os.environ.get("YT_API_KEY"))
    req = yt.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText",
        order="relevance",
    )
    res = req.execute()

    # Extract raw comments
    comments = [
        item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        for item in res.get("items", [])
    ]
    print("Raw Comments Scraped:")
    pprint(comments)

    # Preprocess comments
    cleaned_comments = preprocess_comments(comments)

    # Embed comments
    df_comments = embed_all_comments(cleaned_comments)

    # If a query is provided, find top-k similar comments
    if query:
        return get_top_k_similar_comments(query, df_comments, k=15)

    # If no query is provided, compute similarity scores with a default query
    default_query = "Looking for general insights from comments"
    print("\nNo query provided. Using default query for similarity scoring:")
    print(f"Default Query: {default_query}")
    return get_top_k_similar_comments(default_query, df_comments, k=15)


def get_video_comments(video_url, query=None):
    """Get comments from a youtube video by the videoId"""

    if isinstance(video_url, dict):
        video_url = video_url.get("video_url")

    # Get video id from url
    video_id = get_video_id(video_url)

    # Fetches comments from video
    comments = fetch_comments(video_id, query=query)

    return {"comments": comments}


# App state for the graph (this state flows through the graph from node to node)
class State(TypedDict):
    video_url: str
    comments: str
    response: str
    messages: Annotated[list, add_messages]  # not yet implemented


# Model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# System Prompt/message
prompt = '''
You are an outstandingly helpful YouTube content strategist. Based on a set of YouTube comments, your job is to:

Identify recurring or meaningful themes in the comments (e.g., “motivation,” “career building,” “study support”).

Generate thoughtful video ideas that will resonate with the comment audience under each theme.

### Format Instructions:

Organize the output using “**Focusing on [Theme Name]:**” to group related ideas.

For each idea, provide:

**Title**: A compelling title for the video

**Concept**: A concise explanation of what the video will cover and why it’s valuable

**Possible Segments**: A bullet list of key content or structure that could be included in the video

Also include a section at the end titled:

**General Tips for All Videos:**

A few bullet points with consistent production guidance, such as keeping the video concise or responding to comments.

---

### Example Input:

Comments:

“This course is hard, but I'm proud I made it halfway. 💪”

“So excited to be doing this with other students!”

“Any advice on how to build a quick data portfolio?”

“Can someone explain the case study project?”

“🔥🔥🔥”

“Does anyone else feel stuck in the capstone?”

---

### Example Output:

---

### 🎯 Focusing on **Motivation & Community**

**📌 Title:**  
**_Halfway There: You’re Doing Better Than You Think!_**

**💡 Concept:**  
Uplifting message and mindset tips for students feeling tired mid-course, using real viewer sentiment for emotional resonance.

**📋 Possible Segments:**
- Inspiring quotes from student comments  
- Mental tips to push through the middle of the program  
- Invitation to share personal goals or wins  

---

### 🎯 Focusing on **Career & Portfolio Building**

**📌 Title:**  
**_Build Your First Data Portfolio in 3 Weeks_**

**💡 Concept:**  
Responding to viewer requests for quick, beginner-friendly portfolio tips.

**📋 Possible Segments:**
- Public datasets to start with  
- Project idea walkthroughs (e.g., analyzing YouTube trends)  
- How to showcase results on GitHub  

---

### 🎯 Focusing on **Course Help & Assignments**

**📌 Title:**  
**_Capstone Confusion? Here's What You Need to Know_**

**💡 Concept:**  
Breakdown of the most challenging assignments in the course with step-by-step help.

**📋 Possible Segments:**
- FAQ format with top struggles  
- Visual walkthrough of the case study format  
- Tips for asking questions that get answered  

---

### 💬 General Tips for All Videos

- Keep videos under 10 minutes  
- Use comment quotes as on-screen text overlays  
- Include a question at the end to drive more comments  
- Encourage viewers to like/subscribe/share if they found it helpful  

---

Analyze the following YouTube comments and respond using the exact format above:
{comments}
'''

prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])


def chatbot(state: State):
    print("\nStage 4: Video Idea Generation")
    print("Input Comments for Chatbot:")
    pprint(state["comments"])
    messages_template = prompt_template.invoke(
        {"comments": "\n".join(state["comments"])}
    )
    messages = messages_template.to_messages()
    completion = llm.invoke(messages)
    print("\nGenerated Video Ideas:")
    print(completion.content)
    return {"messages": messages, "response": completion.content}


# Create a new Graph using langchain using the State definition
graph_builder = StateGraph(State)

# Add nodes/functionality or action to the graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("get_video_comments", get_video_comments)

# Set the entry point of the app (Creates the graph sequence)
graph_builder.add_edge(START, "get_video_comments")
graph_builder.add_edge("get_video_comments", "chatbot")

chat_graph = graph_builder.compile()


if __name__ == "__main__":
    # can remain for testing purposes
    result = chat_graph.invoke(
        {
            "video_url": "https://www.youtube.com/watch?v=mQDlCZZsOyo",
            "query": "Looking for video ideas based on comments",
        }
    )

    print("\n=== Final Output ===")
    pprint(result)
