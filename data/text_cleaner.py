import re

def clean_text(text):

    header_patterns = [
        r"^From:.*",
        r"^Subject:.*",
        r"^Organization:.*",
        r"^Lines:.*",
        r"^Path:.*",
        r"^Xref:.*",
        r"^Newsgroups:.*",
        r"^Message-ID:.*",
        r"^Date:.*",
        r"^References:.*",
        r"^Sender:.*",
        r"^Followup-To:.*",
        r"^Distribution:.*",
        r"^Keywords:.*",
        r"^Summary:.*",
        r"^Expires:.*",
        r"^Approved:.*",
        r"^Archive-name:.*",
        r"^Last-modified:.*",
        r"^Version:.*"
    ]

    for pattern in header_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)

    # remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # remove quoted replies
    text = re.sub(r"^>.*", "", text, flags=re.MULTILINE)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()