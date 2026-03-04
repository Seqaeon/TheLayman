import urllib.request
import urllib.parse

def test_url(url):
    print(f"Testing URL: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TheLayman/0.1"})
        with urllib.request.urlopen(req) as resp:
            print(f"Status: {resp.status}")
    except Exception as e:
        print(f"Error: {e}")

# Case 1: Standard urlencode (what I used)
params = {"search_query": "cat:cs.AI OR cat:cs.LG", "max_results": 1}
test_url(f"https://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}")

# Case 2: Manual plus for space, no colon encoding
test_url("https://export.arxiv.org/api/query?search_query=cat:cs.AI+OR+cat:cs.LG&max_results=1")

# Case 3: Manual plus for space, colon encoded
test_url("https://export.arxiv.org/api/query?search_query=cat%3Acs.AI+OR+cat%3Acs.LG&max_results=1")

# Case 4: EXACT URL from server logs
failing_url = "https://export.arxiv.org/api/query?search_query=cat%3Acs.AI+OR+cat%3Acs.LG+OR+cat%3Acs.CL+OR+cat%3Acs.CV&sortBy=submittedDate&sortOrder=desc&max_results=100"
test_url(failing_url)

# Case 5: Parenthesized query
params_p = {"search_query": "(cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV)", "sortBy": "submittedDate", "sortOrder": "desc", "max_results": 100}
test_url(f"https://export.arxiv.org/api/query?{urllib.parse.urlencode(params_p)}")

# Case 6: Multiple categories WITHOUT sortBy
params_no_sort = {"search_query": "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV", "max_results": 100}
test_url(f"https://export.arxiv.org/api/query?{urllib.parse.urlencode(params_no_sort)}")

# Case 7: Single category WITH sortBy
params_s_sort = {"search_query": "cat:cs.AI", "sortBy": "submittedDate", "sortOrder": "desc", "max_results": 1}
test_url(f"https://export.arxiv.org/api/query?{urllib.parse.urlencode(params_s_sort)}")




