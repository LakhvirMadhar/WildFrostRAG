"""One module per wiki-page domain (leaders, stats, charms, ...).

Each module knows how to turn its own page(s) into structured domain
objects, delegating HTTP/caching to scraping/_page_fetching.py and parsing
to data_processing/. No module here should import from another - if two
domains need to share logic, it belongs in _page_fetching.py instead.
"""
