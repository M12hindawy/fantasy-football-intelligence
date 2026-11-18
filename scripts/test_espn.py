import urllib.request, json

urls = [
    ("scoreboard", "http://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"),
    ("standings", "http://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/standings"),
    ("arsenal_schedule", "http://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/teams/359/schedule"),
    ("avl_schedule", "http://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/teams/362/schedule"),
]
for name, url in urls:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
            print(f"ESPN {name}: OK - keys={list(data.keys())[:6]}")
    except Exception as e:
        print(f"ESPN {name}: ERROR - {str(e)[:80]}")
