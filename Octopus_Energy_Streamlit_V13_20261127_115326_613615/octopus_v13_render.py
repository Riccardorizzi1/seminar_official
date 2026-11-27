
import re
import textwrap

from html.parser import HTMLParser

import streamlit as st
from streamlit.delta_generator import DeltaGenerator


# ============================================================
# TYPOGRAPHY
# ============================================================

STYLE = """<style>

[data-testid="stMarkdownContainer"] {
    max-width: none !important;
    width: 100% !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stHtml"] p,
[data-testid="stAlert"] p {

    font-size: 18px !important;
    line-height: 1.65 !important;

    white-space: normal !important;

    max-width: none !important;

    margin: 0 0 .65rem !important;

    overflow-wrap: break-word;

}

[data-testid="stMarkdownContainer"] li {

    font-size: 18px !important;
    line-height: 1.6 !important;

}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {

    font-size: 14px !important;
    line-height: 1.5 !important;

}

[data-testid="stSidebar"]
[data-testid="stMarkdownContainer"] p {

    font-size: 14px !important;
    line-height: 1.45 !important;

}

[data-testid="stMarkdownContainer"] h1 {

    font-size: 2.1rem !important;
    line-height: 1.2 !important;

}

[data-testid="stMarkdownContainer"] h2 {

    font-size: 1.65rem !important;

}

[data-testid="stMarkdownContainer"] h3 {

    font-size: 1.35rem !important;

}

</style>"""


# ============================================================
# NATIVE DASHBOARD HEADER
# ============================================================

HEADER = (

    "# Agile Energy Analytics\n\n"

    "Octopus Energy | Flexibility, "
    "Sustainability and Distributive Justice\n\n"

    "*Research Dashboard — Version 13*"

)


# ============================================================
# PRESERVE STRUCTURED CONTENT
# ============================================================

STRUCT = re.compile(

    r"^\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|"
    r"[>|]|---+\s*$|\[[^]]+\]\([^)]*\)\s*$)"

)

BLOCK = re.compile(

    r"<\s*/?\s*(?:div|p|span|section|h[1-6]|"
    r"br|strong|b|em|i|small)\b",

    re.I

)

SAFE_TAGS = {

    "div", "p", "span", "section",

    "h1", "h2", "h3", "h4", "h5", "h6",

    "br", "strong", "b", "em", "i", "small"

}


# ============================================================
# CONVERT SIMPLE HTML INTO NATIVE MARKDOWN
# ============================================================

class ProseHTML(HTMLParser):

    def __init__(self):

        super().__init__(convert_charrefs=True)

        self.parts = []

    def handle_starttag(self, tag, attrs):

        if tag in {"div", "p", "section"}:

            self.parts.append("\n\n")

        elif re.fullmatch(r"h[1-6]", tag):

            self.parts.append(

                "\n\n" + "#" * int(tag[1]) + " "

            )

        elif tag in {"b", "strong"}:

            self.parts.append("**")

        elif tag in {"i", "em"}:

            self.parts.append("*")

        elif tag == "br":

            self.parts.append(" ")

    def handle_endtag(self, tag):

        if (

            tag in {"div", "p", "section"}

            or re.fullmatch(r"h[1-6]", tag)

        ):

            self.parts.append("\n\n")

        elif tag in {"b", "strong"}:

            self.parts.append("**")

        elif tag in {"i", "em"}:

            self.parts.append("*")

    def handle_data(self, data):

        # Python indentation must not create
        # artificial paragraphs.

        self.parts.append(

            re.sub(r"\s+", " ", data)

        )


# ============================================================
# CONTINUOUS ACADEMIC PARAGRAPHS
# ============================================================

def reflow(text):

    s = textwrap.dedent(text).strip()

    if not s:

        return s

    # Preserve real programming code, mathematics,
    # scripts and CSS.

    if re.search(

        r"```|~~~|\$\$|\\\[|\\\]|"
        r"<(?:style|script|pre|textarea)\b",

        s,

        re.I

    ):

        return text

    # Convert simple HTML explanations to Markdown.

    if BLOCK.search(s):

        tags = set(

            re.findall(

                r"<\s*/?\s*([a-zA-Z][\w-]*)\b",

                s.lower()

            )

        )

        if not tags <= SAFE_TAGS:

            return text

        parser = ProseHTML()

        parser.feed(s)

        parser.close()

        s = "".join(parser.parts).strip()

    if not re.search(

        r"\n|<br\s*/?>",

        s,

        re.I

    ):

        return s

    # Preserve actual lists, tables and quotations.

    if any(

        re.match(

            r"^\s*(?:[|>]|[-*+]\s|\d+[.)]\s)",

            line

        )

        for line in s.splitlines()

    ):

        return s

    # Preserve references and clickable sources.

    if re.search(r"https?://|\]\(", s):

        return s

    lines = s.splitlines()

    blocks = []

    pending = []

    def flush():

        if not pending:

            return

        value = " ".join(pending)

        # Split only at complete sentence boundaries.

        sentences = re.split(

            r"(?<=[.!?])\s+(?=[A-Z])",

            value

        )

        group = []

        for sentence in sentences:

            # Start a new paragraph when the argument
            # changes substantially.

            transition = re.match(

                r"^(?:However,|Nevertheless,|By contrast,|"
                r"From a distributive justice perspective,)",

                sentence

            )

            if (

                group

                and transition

                and sum(

                    len(x.split())

                    for x in group

                ) >= 20

            ):

                blocks.append(" ".join(group))

                group = []

            group.append(sentence)

            if sum(

                len(x.split())

                for x in group

            ) >= 80:

                blocks.append(" ".join(group))

                group = []

        if group:

            blocks.append(" ".join(group))

        pending.clear()

    for line in lines:

        line = line.strip()

        # Ignore artificial blank lines inherited
        # from Python source formatting.

        if not line:

            continue

        is_market_heading = re.fullmatch(

            r"\*\*What the electricity market tells us\*\*",

            line,

            re.I

        )

        if STRUCT.match(line) or is_market_heading:

            flush()

            blocks.append(line)

        else:

            pending.append(line)

    flush()

    return "\n\n".join(blocks)


# ============================================================
# IDENTIFY THE BROKEN HEADER
# ============================================================

def is_header(s):

    return (

        isinstance(s, str)

        and "Agile Energy Analytics" in s

        and bool(

            re.search(

                r"<\s*h1\b|Research Dashboard",

                s,

                re.I

            )

        )

    )


# ============================================================
# STREAMLIT RENDERING INTERCEPTION
# ============================================================

def renderer(fn, name):

    target = getattr(fn, "__self__", None)

    # Do not modify Pandas, file operations or other
    # non-Streamlit objects.

    if not isinstance(target, DeltaGenerator):

        return fn

    def call(*args, **kwargs):

        a = list(args)

        k = dict(kwargs)

        key = next(

            (

                x

                for x in ("body", "text")

                if x in k

            ),

            None

        )

        value = a[0] if a else k.get(key)

        # Replace the problematic dashboard header
        # before Streamlit can display it as code.

        if is_header(value):

            # Preserve any existing CSS embedded
            # in the old header.

            for css in re.findall(

                r"<style\b[^>]*>.*?</style>",

                value,

                re.I | re.S

            ):

                target.markdown(

                    css,

                    unsafe_allow_html=True

                )

            return target.markdown(HEADER)

        # Preserve legitimate programming examples.

        if name == "code":

            return fn(*args, **kwargs)

        if name == "write":

            a = [

                reflow(x)

                if isinstance(x, str)

                else x

                for x in a

            ]

        elif isinstance(value, str):

            fixed = reflow(value)

            if (

                name in ("html", "text")

                and fixed != value

                and not re.search(

                    r"<[a-zA-Z][\w-]*\b",

                    fixed

                )

            ):

                return target.markdown(fixed)

            if a:

                a[0] = fixed

            elif key:

                k[key] = fixed

        return fn(*a, **k)

    return call
