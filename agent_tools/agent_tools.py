


def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")


def search_wikipedia(query):
    """Searches Wikipedia and returns the summary of the first result."""
    from wikipedia import summary

    try:
        # Limit to two sentences for brevity
        return summary(query, sentences=2)
    except:
        return "I couldn't find any information on that."
    

# def use_tavily_search(query):
#     """Uses Tavily to search for the query and return the results."""
#     tavily = TavilySearch(max_results=5,topic="general"),
#     results = tavil. .search(query)
#     return results

