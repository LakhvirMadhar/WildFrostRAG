"""The filename rule for a wiki page's cached HTML.

Lives in domain/ (the bottom layer) because both repositories/ and
scraping/ need it, and the import-linter layers contract doesn't let those
two import each other.
"""

import re

_INVALID_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|]')


def wiki_page_filename(name: str) -> str:
    """Return the cached-HTML filename for a wiki entity, e.g. "What Is This?" -> "What Is This.html".

    Strips characters Windows and POSIX filesystems reject; spaces are kept.
    """
    return f"{_INVALID_FILENAME_CHARS.sub('', name)}.html"
