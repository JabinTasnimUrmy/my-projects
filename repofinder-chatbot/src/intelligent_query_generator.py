# src/intelligent_query_generator.py
import json
import re  
from config import GROQ_API_KEY
from groq import Groq

if not GROQ_API_KEY:
    raise ValueError("Groq API key not found. Please set GROQ_API_KEY in your .env file.")

client = Groq(api_key=GROQ_API_KEY)

def generate_search_queries(project_description: str, count: int = 3) -> list[str]:
    """
    Uses the Llama 3 model to summarize long text and generate technical GitHub search queries.
    """
    system_prompt = f"""
    You are an expert software engineer and GitHub search specialist. Your task is to analyze a user's project description and convert it into a Python list of {count} effective GitHub search queries.

    **Analysis Steps:**
    1.  Read the entire description provided by the user.
    2.  Identify the core technologies, languages, and libraries mentioned (e.g., Python, OpenCV, YOLOv8).
    3.  Identify the main problem or goal (e.g., real-time object detection, safety compliance, video analysis).
    4.  Synthesize these concepts into short, technical search queries a developer would use.
    5.  Omit conversational phrases and non-technical details.

    **Output Format:**
    Return ONLY the Python-parsable list of strings. Do not add any extra text, explanation, or markdown.

    **Example for a long, complex description:**
    User provides a detailed project specification for a real-time safety compliance system. The system must use factory CCTV camera feeds, run an object detection model like YOLOv8 to identify workers, and then run a second classification model on each detected worker to check for personal protective equipment (PPE) like hard hats and safety vests.
    Your response: ["real-time ppe detection yolov8", "cctv safety compliance monitoring", "hard hat detection computer vision python"]
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": project_description},
            ],
            model="llama3-8b-8192",
            temperature=0.2,
        )
        
        model_answer = chat_completion.choices[0].message.content.strip()
        
        # MAKE PARSING MORE ROBUST
        # Instead of a strict check, find the start and end of the list within the text.
        list_start_index = model_answer.find('[')
        list_end_index = model_answer.rfind(']') + 1

        if list_start_index != -1 and list_end_index != 0:
            query_list_str = model_answer[list_start_index:list_end_index]
            queries = json.loads(query_list_str)
            if isinstance(queries, list) and queries:
                return queries
        

        # If the response is still invalid after robust parsing, handle it gracefully
        print(f"--> Model generated an invalid response that could not be parsed: {model_answer}")
        return [ " ".join(re.findall(r'\w+', model_answer)) ] if model_answer else [project_description]

    except Exception as e:
        print(f"\nAn error occurred: {e}") # This will now report the original error correctly
        return [project_description]