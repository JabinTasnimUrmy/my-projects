# src/chatbot_logic.py
from src.github_client import search_repositories
from src.semantic_analyzer import find_most_similar_project
from src.intelligent_query_generator import generate_search_queries 

def start_chat():
    """Main function to run the chatbot's interactive loop."""
    print("\n--- Intelligent RepoFinder Chatbot ---")
    print("Welcome! Describe your project idea, and I'll translate it into a smart search.")
    print("Type 'quit' or 'exit' to leave.")

    while True:
        user_query = input("\nYour project idea: ").strip()

        if user_query.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if not user_query:
            print("Please provide a description of your idea.")
            continue
        
        
        print("\nTranslating your idea into technical search queries...")
        smart_queries = generate_search_queries(user_query)
        print(f"Generated queries: {smart_queries}")
        
        all_repositories = []
        # Search GitHub with each of the smart queries to get a wide range of results
        for query in smart_queries:
            print(f"Searching GitHub for: '{query}'...")
            # We extend the list with results from each query
            results = search_repositories(query)
            if results:
                all_repositories.extend(results)
        
        # Remove duplicate repositories that might appear from different queries
        unique_repositories = list({repo['id']: repo for repo in all_repositories}.values())
        print(f"Found {len(unique_repositories)} unique potential projects.")
        
        
        if not unique_repositories:
            print("Could not find any relevant projects, even with smart search. Please try rephrasing your idea.")
            continue
        
        print("Analyzing results for the best semantic match...")
        # Detailed user query to find the best match in our rich list
        best_match = find_most_similar_project(user_query, unique_repositories)

        if best_match:
            print("\n✅ Found a project that seems like a great match!")
            print(f"   Project: {best_match['full_name']}")
            print(f"   Description: {best_match.get('description')}")
            print(f"   ⭐ Stars: {best_match.get('stargazers_count', 0)}")
            print(f"   Language: {best_match.get('language')}")
            print(f"   🔗 URL: {best_match['html_url']}")
        else:
            print("\n💡 Your idea seems to be quite unique!")
            print("I couldn't find a high-confidence match.")
            closest_idea = unique_repositories[0]
            print("The most closely related project I found based on keywords is:")
            print(f"   Project: {closest_idea['full_name']}")
            print(f"   Description: {closest_idea.get('description')}")
            print(f"   🔗 URL: {closest_idea['html_url']}")