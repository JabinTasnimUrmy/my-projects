# app.py
import webbrowser
import threading
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from src.intelligent_query_generator import generate_search_queries
from src.github_client import search_repositories
from src.semantic_analyzer import find_most_similar_project

app = Flask(__name__)
CORS(app) 

def get_repo_suggestion(user_query):
    print(f"Received query: {user_query}")
    
    smart_queries = generate_search_queries(user_query)
    print(f"Generated queries: {smart_queries}")
    
    all_repositories = []
    for query in smart_queries:
        results = search_repositories(query)
        if results:
            all_repositories.extend(results)
    
    if not all_repositories:
        return {"type": "error", "message": "Could not find any projects matching the generated queries."}
        
    unique_repositories = list({repo['id']: repo for repo in all_repositories}.values())
    
    best_match = find_most_similar_project(user_query, unique_repositories)
    
    if best_match:
        return {
            "type": "success",
            "unique": False,
            "project": {
                "name": best_match['full_name'],
                "description": best_match.get('description'),
                "stars": best_match.get('stargazers_count', 0),
                "language": best_match.get('language'),
                "url": best_match['html_url']
            }
        }
    else:
        closest_idea = unique_repositories[0]
        return {
            "type": "success",
            "unique": True,
            "project": {
                "name": closest_idea['full_name'],
                "description": closest_idea.get('description'),
                "url": closest_idea['html_url']
            }
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_suggestion', methods=['POST'])
def get_suggestion_api():
    try:
        user_query = request.json.get('idea')
        if not user_query:
            return jsonify({"error": "No project idea provided."}), 400

        result = get_repo_suggestion(user_query)
        
        return jsonify(result)
    except Exception as e:
        print(f"An error occurred in the API route: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500


if __name__ == '__main__':
    # Define the URL where the app will be running
    host = "127.0.0.1"
    port = 5000
    url = f"http://{host}:{port}"

    # This function will be called after a short delay to open the browser
    def open_browser():
        webbrowser.open_new(url)

    # Start a timer that will call the open_browser function after 1 second
    # This gives the Flask server a moment to start up before the browser tries to connect
    threading.Timer(1, open_browser).start()
    
    # Run the Flask web server
    app.run(host=host, port=port, debug=True)