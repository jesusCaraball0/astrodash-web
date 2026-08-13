"""
Download public supernova spectra added to WISeREP during an inclusive date range.

Output layout:
    OUTPUT/
      spectra/
        <IAU>_<WISeREP_SPEC_ID>.<original_ascii_extension>
        ...
      metadata.csv

metadata.csv columns:
    iau,filename,type,redshift

The script:
  * searches WISeREP's *spectra* search, not object search;
  * filters by spectrum Creation Date (UT), not observation date;
  * keeps public SN/SLSN spectra only;
  * downloads only the matching ASCII spectrum files;
  * uses stable filenames based on IAU name + WISeREP spectrum ID;
  * is idempotent and can append multiple monthly runs into the same OUTPUT;
  * keeps no raw export or temporary duplicate dataset.

Dependencies:
    pip install requests beautifulsoup4

Example:
    python wiserep_monthly_scrape.py \
        --start 2026-07-01 \
        --end 2026-07-31 \
        --output ./wiserep_data
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html as html_lib
import io
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://www.wiserep.org"
SEARCH_URL = f"{BASE_URL}/search/spectra"

# WISeREP's public community tooling accesses pages with a TNS-style marker
# User-Agent. A descriptive marker is used here; no API key or login is needed
# for public data.
USER_AGENT = (
    'tns_marker{"tns_id":0,"type":"bot","name":"AstroDASH_WISeREP_Ingest"}'
)

METADATA_COLUMNS = ["iau", "filename", "type", "redshift"]

DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
FLOAT_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)

KNOWN_ASCII_SUFFIXES = {
    ".txt",
    ".dat",
    ".ascii",
    ".csv",
    ".lnw",
    ".flm",
    ".spec",
}


@dataclass(frozen=True)
class CandidateSpectrum:
    iau: str
    spec_id: str
    obj_id: str
    creation_date: date
    row_type: str
    row_redshift: str
    ascii_file: str


@dataclass(frozen=True)
class ObjectPage:
    html: str
    sn_type: str
    redshift: str


@dataclass(frozen=True)
class AsciiLink:
    url: str
    filename: str
    row_text: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download public WISeREP SN spectra whose spectrum Creation Date "
            "falls in an inclusive YYYY-MM-DD range."
        )
    )
    p.add_argument("--start", required=True, help="Inclusive start date, YYYY-MM-DD")
    p.add_argument("--end", required=True, help="Inclusive end date, YYYY-MM-DD")
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Dataset directory containing spectra/ and metadata.csv",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=1000,
        help="Safety cap on WISeREP search-result pages (default: 1000)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds (default: 60)",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=0.10,
        help="Delay between WISeREP requests in seconds (default: 0.10)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files that already exist",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra diagnostics",
    )
    return p.parse_args()


def parse_cli_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}; expected YYYY-MM-DD"
        ) from exc


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/csv;q=0.9,*/*;q=0.8",
        }
    )

    retries = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=8, pool_maxsize=8)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def request(
    session: requests.Session,
    url: str,
    *,
    timeout: float,
    delay: float,
    params=None,
) -> requests.Response:
    if delay > 0:
        time.sleep(delay)

    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response


def compact(text: object) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", compact(name).lower())


def normalize_iau(value: str) -> str:
    value = compact(value)
    value = re.sub(r"^SN\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", "", value)
    return value


def safe_token(value: str) -> str:
    value = compact(value)
    value = re.sub(r"[^A-Za-z0-9._+-]+", "_", value)
    value = value.strip("._")
    return value or "unknown"


def parse_date_from_text(value: str) -> date | None:
    m = DATE_RE.search(compact(value))
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def normalize_redshift(value: object) -> str:
    """
    Return a clean numeric redshift string, or "" when unavailable.

    We intentionally do NOT convert missing redshift to 0.0.
    """
    text = compact(value)
    if not text or text.lower() in {"nan", "none", "null", "-", "n/a", "na"}:
        return ""

    m = FLOAT_RE.search(text)
    if not m:
        return ""

    try:
        z = float(m.group(0))
    except ValueError:
        return ""

    # Stable compact representation without introducing fake precision.
    return format(z, ".12g")


def find_column(
    fieldnames: Iterable[str],
    preferred: Iterable[str],
    contains: Iterable[str] = (),
) -> str | None:
    fields = list(fieldnames)
    normalized = {normalize_col(f): f for f in fields}

    for alias in preferred:
        key = normalize_col(alias)
        if key in normalized:
            return normalized[key]

    contains_norm = [normalize_col(x) for x in contains]
    for f in fields:
        nf = normalize_col(f)
        if all(token in nf for token in contains_norm):
            return f
    return None


def clean_csv_text(text: str) -> str:
    # Strip UTF-8 BOM and leading blank lines that occasionally confuse DictReader.
    return text.lstrip("\ufeff\r\n ")


def parse_csv_rows(text: str) -> list[dict[str, str]]:
    cleaned = clean_csv_text(text)
    if not cleaned:
        return []

    reader = csv.DictReader(io.StringIO(cleaned, newline=""))
    if not reader.fieldnames:
        return []

    return [
        {compact(k): compact(v) for k, v in row.items() if k is not None}
        for row in reader
    ]


def parse_html_table_rows(text: str) -> list[dict[str, str]]:
    """
    Fallback parser if WISeREP returns HTML rather than CSV.

    Finds the table whose headers contain Spec. ID and Creation Date.
    """
    soup = BeautifulSoup(text, "html.parser")

    for table in soup.find_all("table"):
        header_cells = table.find_all("th")
        headers = [compact(th.get_text(" ", strip=True)) for th in header_cells]
        norm_headers = [normalize_col(h) for h in headers]

        if not headers:
            continue
        if not any("specid" == h or "spectrumid" == h for h in norm_headers):
            continue
        if not any("creationdate" in h for h in norm_headers):
            continue

        rows: list[dict[str, str]] = []
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if not cells:
                continue

            values = [compact(td.get_text(" ", strip=True)) for td in cells]
            if len(values) < len(headers):
                continue

            row = dict(zip(headers, values[: len(headers)]))

            # Preserve the displayed ASCII filename if it is an href rather than text.
            for td, header in zip(cells, headers):
                if "ascii" not in normalize_col(header):
                    continue
                a = td.find("a", href=True)
                if a:
                    row[header] = compact(a.get_text(" ", strip=True)) or a["href"]

            rows.append(row)
        return rows

    return []


def form_control_context(tag) -> str:
    """Gather only *nearby* form text, avoiding the entire form as context."""
    chunks = []
    node = tag
    for _ in range(3):  # tag + parent + grandparent only
        if node is None:
            break
        try:
            chunks.append(compact(node.get_text(" ", strip=True)))
        except Exception:
            pass
        node = node.parent
    return " ".join(chunks).lower()


def discover_date_filter_params(
    session: requests.Session,
    start: date,
    end: date,
    *,
    timeout: float,
    delay: float,
    verbose: bool,
) -> list[tuple[str, str]]:
    """
    Discover the current WISeREP spectra-search Creation Date input names.

    This avoids hard-coding undocumented form-field names. If discovery fails,
    the caller falls back to walking results sorted by Creation Date and filtering
    locally.
    """
    try:
        response = request(
            session, SEARCH_URL, timeout=timeout, delay=delay, params={"format": "html"}
        )
    except requests.RequestException as exc:
        if verbose:
            print(f"[date-filter] Could not inspect search form: {exc}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    form = None

    for candidate in soup.find_all("form"):
        action = compact(candidate.get("action"))
        if "search/spectra" in action:
            form = candidate
            break
    if form is None:
        form = soup.find("form")
    if form is None:
        return []

    # First preference: field names containing "creation". This is much safer
    # than broad nearby-text matching because the form also contains observation
    # and last-modified date ranges.
    candidates = []
    for inp in form.find_all("input"):
        name = compact(inp.get("name"))
        if not name:
            continue
        lname = name.lower()
        if "display" in lname or "modifier" in lname or "createdby" in lname:
            continue
        if "creation" in lname:
            candidates.append(inp)

    # If WISeREP renames the fields, fall back to narrowly scoped label/context
    # matching.
    if len(candidates) < 2:
        candidates = []
        for inp in form.find_all("input"):
            name = compact(inp.get("name"))
            if not name:
                continue
            lname = name.lower()
            if "display" in lname or "modifier" in lname or "createdby" in lname:
                continue
            context = form_control_context(inp)
            if "date" in lname and "creation date" in context and "last modified" not in context:
                candidates.append(inp)

    # De-duplicate identical tag objects while preserving DOM order.
    unique = []
    seen_ids = set()
    for tag in candidates:
        ident = id(tag)
        if ident not in seen_ids:
            unique.append(tag)
            seen_ids.add(ident)
    candidates = unique

    if len(candidates) < 2:
        if verbose:
            names = [compact(x.get("name")) for x in candidates]
            print(f"[date-filter] Could not confidently find two date inputs: {names}")
        return []

    def direction(tag) -> str:
        name = compact(tag.get("name")).lower()
        context = form_control_context(tag)
        text = f"{name} {context}"
        if any(k in text for k in ("from", "start", "min", "begin", "after")):
            return "start"
        if any(k in text for k in (" to ", "end", "max", "until", "before")):
            return "end"
        return ""

    start_tag = next((t for t in candidates if direction(t) == "start"), None)
    end_tag = next((t for t in candidates if direction(t) == "end"), None)

    if start_tag is None or end_tag is None:
        # WISeREP presents these as an ordered date pair; use DOM order.
        start_tag, end_tag = candidates[0], candidates[1]

    start_name = compact(start_tag.get("name"))
    end_name = compact(end_tag.get("name"))
    if not start_name or not end_name:
        return []

    items = [
        (start_name, start.isoformat()),
        (end_name, end.isoformat()),
    ]

    # If both range endpoints use the same field name, repeated query keys are
    # intentional and requests will serialize both values.
    if verbose:
        print(f"[date-filter] Using WISeREP fields: {items}")

    return items


def public_filter_params_from_form(
    session: requests.Session,
    *,
    timeout: float,
    delay: float,
    verbose: bool,
) -> list[tuple[str, str]]:
    """
    Best-effort discovery of the spectra-search Public=Yes control.
    Unauthenticated WISeREP only permits public downloads anyway, but adding the
    filter reduces irrelevant rows if the UI exposes private metadata.
    """
    try:
        response = request(
            session, SEARCH_URL, timeout=timeout, delay=delay, params={"format": "html"}
        )
    except requests.RequestException:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    for select in soup.find_all("select"):
        name = compact(select.get("name"))
        if not name:
            continue
        context = form_control_context(select)
        if "public" not in name.lower() and "public" not in context:
            continue

        for option in select.find_all("option"):
            if compact(option.get_text(" ", strip=True)).lower() == "yes":
                value = compact(option.get("value"))
                if value:
                    if verbose:
                        print(f"[public-filter] Using {name}={value}")
                    return [(name, value)]

    return []


def fetch_search_page(
    session: requests.Session,
    base_query: list[tuple[str, str]],
    page: int,
    *,
    timeout: float,
    delay: float,
    verbose: bool,
) -> list[dict[str, str]]:
    query = list(base_query)
    query.extend(
        [
            ("format", "csv"),
            ("order", "creationdate"),
            ("sort", "desc"),
            ("page", str(page)),
        ]
    )

    response = request(
        session, SEARCH_URL, timeout=timeout, delay=delay, params=query
    )

    prefix = response.text.lstrip()[:500].lower()
    looks_html = (
        prefix.startswith("<!doctype")
        or prefix.startswith("<html")
        or "<html" in prefix
    )

    # CSV is the preferred documented metadata export. If WISeREP returns its
    # HTML UI despite format=csv, parse the result table instead.
    if looks_html:
        rows = parse_html_table_rows(response.text)
    else:
        rows = parse_csv_rows(response.text)

    # Some deployments may return CSV with a reduced display-column set. If the
    # creation date is absent, retry the same page as HTML before giving up.
    if rows and not find_column(
        rows[0].keys(),
        ("Creation Date (UT)", "Creation Date", "Creation date (UT)", "Creation date"),
        ("creation", "date"),
    ):
        html_query = [(k, v) for k, v in query if k != "format"]
        html_query.append(("format", "html"))
        html_response = request(
            session, SEARCH_URL, timeout=timeout, delay=delay, params=html_query
        )
        html_rows = parse_html_table_rows(html_response.text)
        if html_rows:
            rows = html_rows
            response = html_response

    if verbose:
        print(f"[search] page={page} rows={len(rows)} url={response.url}")

    return rows


def identify_search_columns(rows: list[dict[str, str]]) -> dict[str, str | None]:
    if not rows:
        return {}

    fields = list(rows[0].keys())

    columns = {
        "creation": find_column(
            fields,
            (
                "Creation Date (UT)",
                "Creation Date",
                "Creation date (UT)",
                "Creation date",
            ),
            ("creation", "date"),
        ),
        "iau": find_column(
            fields,
            ("Obj. IAU Name", "Obj IAU Name", "IAU Name", "IAUName"),
            ("iau", "name"),
        ),
        "obj_id": find_column(
            fields,
            ("Obj. ID", "Obj ID", "Object ID"),
            ("obj", "id"),
        ),
        "spec_id": find_column(
            fields,
            ("Spec. ID", "Spec ID", "Spectrum ID"),
            ("spec", "id"),
        ),
        "type": find_column(
            fields,
            ("Obj. Type", "Obj Type", "Type", "Object Type", "Obj Family Type"),
        ),
        "redshift": find_column(
            fields,
            ("Redshift", "Obj. Redshift", "Obj Redshift", "Host Redshift"),
        ),
        "ascii": find_column(
            fields,
            (
                "Spectrum ascii File",
                "Spectrum ASCII File",
                "ASCII File",
                "Ascii File",
            ),
            ("ascii",),
        ),
        "public": find_column(fields, ("Public",)),
    }

    return columns


def collect_candidates(
    session: requests.Session,
    start: date,
    end: date,
    *,
    max_pages: int,
    timeout: float,
    delay: float,
    verbose: bool,
) -> list[CandidateSpectrum]:
    date_params = discover_date_filter_params(
        session, start, end, timeout=timeout, delay=delay, verbose=verbose
    )
    public_params = public_filter_params_from_form(
        session, timeout=timeout, delay=delay, verbose=verbose
    )

    base_query = date_params + public_params
    server_date_filter = bool(date_params)

    candidates: list[CandidateSpectrum] = []
    seen_search_rows: set[tuple[str, str, str]] = set()
    previous_page_signature = None
    columns = None

    for page in range(max_pages):
        rows = fetch_search_page(
            session,
            base_query,
            page,
            timeout=timeout,
            delay=delay,
            verbose=verbose,
        )
        if not rows:
            break

        if columns is None:
            columns = identify_search_columns(rows)
            if not columns.get("creation"):
                raise RuntimeError(
                    "WISeREP search results no longer expose a Creation Date column. "
                    "Cannot guarantee upload-date filtering safely."
                )
            if not columns.get("iau"):
                raise RuntimeError(
                    "WISeREP search results no longer expose an IAU-name column."
                )
            if not columns.get("spec_id") and not columns.get("ascii"):
                raise RuntimeError(
                    "WISeREP search results expose neither Spec. ID nor ASCII filename; "
                    "cannot reliably match physical spectra."
                )

            if verbose:
                print(f"[search] detected columns: {columns}")

        # Detect a server that ignores page= and keeps returning the same page.
        signature_parts = []
        for row in rows[:10]:
            signature_parts.append(
                (
                    compact(row.get(columns.get("spec_id") or "", "")),
                    compact(row.get(columns.get("iau") or "", "")),
                    compact(row.get(columns.get("creation") or "", "")),
                )
            )
        page_signature = tuple(signature_parts)
        if page > 0 and page_signature == previous_page_signature:
            if verbose:
                print("[search] repeated page detected; stopping pagination")
            break
        previous_page_signature = page_signature

        page_dates: list[date] = []

        for row in rows:
            created = parse_date_from_text(row.get(columns["creation"], ""))
            if created is None:
                continue
            page_dates.append(created)

            # Always verify the requested interval locally even when the server
            # date filter was successfully discovered.
            if created < start or created > end:
                continue

            public_col = columns.get("public")
            if public_col:
                public_text = compact(row.get(public_col, "")).lower()
                if public_text in {"no", "false", "0", "private"}:
                    continue

            iau = normalize_iau(row.get(columns["iau"], ""))
            if not iau:
                continue

            spec_id = compact(row.get(columns.get("spec_id") or "", ""))
            obj_id = compact(row.get(columns.get("obj_id") or "", ""))
            row_type = compact(row.get(columns.get("type") or "", ""))
            row_redshift = compact(row.get(columns.get("redshift") or "", ""))
            ascii_file = compact(row.get(columns.get("ascii") or "", ""))

            dedupe_key = (iau, spec_id, ascii_file)
            if dedupe_key in seen_search_rows:
                continue
            seen_search_rows.add(dedupe_key)

            candidates.append(
                CandidateSpectrum(
                    iau=iau,
                    spec_id=spec_id,
                    obj_id=obj_id,
                    creation_date=created,
                    row_type=row_type,
                    row_redshift=row_redshift,
                    ascii_file=ascii_file,
                )
            )

        # If server-side date filtering worked, empty-of-range pages should not
        # happen, and normal pagination ends on an empty page.
        if server_date_filter:
            continue

        # Fallback mode walks newest -> oldest. Once an entire page is older
        # than START, no later page can contain a match.
        if page_dates and max(page_dates) < start:
            break

    else:
        raise RuntimeError(
            f"Reached --max-pages={max_pages} before proving the search was complete."
        )

    return candidates


def extract_property_from_object_html(text: str, label: str) -> str:
    escaped = re.escape(label)
    patterns = [
        rf'{escaped}</span><div class="value"><b>([^<]*)',
        rf'{escaped}</span><b><div class="value">([^<]*)',
        rf'{escaped}</span>\s*<div class="value">\s*<b>([^<]*)',
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return compact(html_lib.unescape(m.group(1)))
    return ""


def infer_type_from_object_html(text: str) -> str:
    sn_type = extract_property_from_object_html(text, "Type")

    # Match the behavior used by the community wiserep_api package: if the
    # object's main type is only "SN", use a more specific classification from
    # the classification-report table when one is available.
    if sn_type and sn_type != "SN":
        return sn_type

    soup = BeautifulSoup(text, "html.parser")
    for cell in soup.select(".cell-objtype_name"):
        candidate = compact(cell.get_text(" ", strip=True))
        if candidate and candidate != "SN":
            return candidate

    return sn_type or "Unknown"


def fetch_object_page(
    session: requests.Session,
    iau: str,
    obj_id: str,
    *,
    timeout: float,
    delay: float,
) -> ObjectPage:
    urls = []
    if obj_id:
        urls.append(f"{BASE_URL}/object/{quote(obj_id, safe='')}")
    urls.append(f"{BASE_URL}/iauname/{quote(iau, safe='+-.')}")

    last_error = None
    for url in urls:
        try:
            response = request(
                session, url, timeout=timeout, delay=delay
            )
        except requests.RequestException as exc:
            last_error = exc
            continue

        text = response.text
        sn_type = infer_type_from_object_html(text)
        redshift = normalize_redshift(
            extract_property_from_object_html(text, "Redshift")
        )
        return ObjectPage(html=text, sn_type=sn_type, redshift=redshift)

    raise RuntimeError(
        f"Could not load WISeREP object page for {iau}: {last_error}"
    )


def normalize_download_url(value: str) -> str:
    value = html_lib.unescape(unquote(compact(value)))
    if value.startswith("//"):
        value = "https:" + value
    if value.startswith("www."):
        value = "https://" + value
    if value.startswith("http://"):
        # Prefer TLS on the modern site; download() below has an HTTP fallback.
        value = "https://" + value[len("http://") :]
    return value


def extract_ascii_links(object_html: str) -> list[AsciiLink]:
    soup = BeautifulSoup(object_html, "html.parser")
    found: dict[str, AsciiLink] = {}

    for a in soup.find_all("a", href=True):
        href = html_lib.unescape(compact(a.get("href")))
        if "asciifile=" not in href:
            continue

        absolute_href = urljoin(BASE_URL, href)
        query = parse_qs(urlparse(absolute_href).query)
        values = query.get("asciifile", [])

        # Some WISeREP links historically use partially encoded
        # "https%3A//..." values. parse_qs normally handles these; regex is a
        # fallback for malformed query strings.
        if not values:
            m = re.search(r"asciifile=([^&\"<>]+)", href)
            if m:
                values = [m.group(1)]

        for raw in values:
            url = normalize_download_url(raw)
            if not url.startswith(("http://", "https://")):
                continue

            filename = Path(urlparse(url).path).name
            if not filename:
                continue

            tr = a.find_parent("tr")
            row_text = compact(tr.get_text(" ", strip=True)) if tr else ""
            found[url] = AsciiLink(url=url, filename=filename, row_text=row_text)

    # Fallback modeled on the current community wiserep_api parser.
    for m in re.finditer(r"asciifile=(https?%3A//[^\"&<]+)", object_html):
        url = normalize_download_url(m.group(1))
        filename = Path(urlparse(url).path).name
        if url.startswith(("http://", "https://")) and filename:
            found.setdefault(
                url, AsciiLink(url=url, filename=filename, row_text="")
            )

    return list(found.values())


def ascii_filename_from_search_value(value: str) -> str:
    value = html_lib.unescape(compact(value))
    if not value:
        return ""

    # Direct URL
    if value.startswith(("http://", "https://")):
        return Path(urlparse(value).path).name

    # Download href with asciifile query parameter
    if "asciifile=" in value:
        absolute = urljoin(BASE_URL, value)
        query = parse_qs(urlparse(absolute).query)
        vals = query.get("asciifile", [])
        if vals:
            return Path(urlparse(normalize_download_url(vals[0])).path).name

    # Plain displayed filename
    return Path(value).name


def resolve_ascii_link(
    candidate: CandidateSpectrum,
    links: list[AsciiLink],
) -> AsciiLink | None:
    search_filename = ascii_filename_from_search_value(candidate.ascii_file)

    if search_filename:
        exact = [x for x in links if x.filename == search_filename]
        if len(exact) == 1:
            return exact[0]

        # Case-insensitive fallback for servers/filesystems that changed case.
        ci = [x for x in links if x.filename.lower() == search_filename.lower()]
        if len(ci) == 1:
            return ci[0]

    if candidate.spec_id:
        spec_matches = [
            x
            for x in links
            if re.search(
                rf"(?<!\d){re.escape(candidate.spec_id)}(?!\d)", x.row_text
            )
        ]
        if len(spec_matches) == 1:
            return spec_matches[0]

    return None


def is_supernova_type(sn_type: str) -> bool:
    t = compact(sn_type).upper()
    return t.startswith("SN") or t.startswith("SLSN")


def choose_output_suffix(download_filename: str) -> str:
    suffix = Path(download_filename).suffix.lower()
    if suffix in KNOWN_ASCII_SUFFIXES:
        return suffix
    # It is an "asciifile" download, so use .txt for unknown/no extensions.
    return ".txt"


def stable_output_filename(
    iau: str,
    spec_id: str,
    download_url: str,
    download_filename: str,
) -> str:
    iau_token = safe_token(iau)

    if spec_id:
        spec_token = safe_token(spec_id)
    else:
        # Stable fallback if WISeREP ever omits Spec. ID from search output.
        spec_token = hashlib.sha1(download_url.encode("utf-8")).hexdigest()[:12]

    suffix = choose_output_suffix(download_filename)
    return f"{iau_token}_{spec_token}{suffix}"


def download_ascii(
    session: requests.Session,
    url: str,
    destination: Path,
    *,
    timeout: float,
    delay: float,
) -> None:
    attempts = [url]
    if url.startswith("https://"):
        attempts.append("http://" + url[len("https://") :])

    last_error = None
    for attempt in attempts:
        try:
            response = request(
                session, attempt, timeout=timeout, delay=delay
            )
        except requests.RequestException as exc:
            last_error = exc
            continue

        content = response.content
        if not content:
            last_error = RuntimeError("empty response")
            continue

        # Protect against silently saving an HTML error/login page as a spectrum.
        prefix = content[:512].lstrip().lower()
        ctype = response.headers.get("Content-Type", "").lower()
        if (
            ("text/html" in ctype or prefix.startswith(b"<"))
            and (b"<html" in prefix or b"<!doctype" in prefix)
        ):
            last_error = RuntimeError("received HTML instead of spectrum data")
            continue

        destination.write_bytes(content)
        return

    raise RuntimeError(f"Failed to download {url}: {last_error}")


def read_existing_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}

    rows: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = compact(row.get("filename", ""))
            if not filename:
                continue
            rows[filename] = {
                "iau": compact(row.get("iau", "")),
                "filename": filename,
                "type": compact(row.get("type", "")),
                "redshift": compact(row.get("redshift", "")),
            }
    return rows


def write_metadata_atomic(
    path: Path,
    rows_by_filename: dict[str, dict[str, str]],
) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")

    rows = list(rows_by_filename.values())
    rows.sort(key=lambda r: (r["iau"].lower(), r["filename"].lower()))

    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    tmp.replace(path)


def main() -> int:
    args = parse_args()

    start = parse_cli_date(args.start)
    end = parse_cli_date(args.end)
    if start > end:
        print("--start must be <= --end", file=sys.stderr)
        return 2

    output_dir: Path = args.output.expanduser().resolve()
    spectra_dir = output_dir / "spectra"
    metadata_path = output_dir / "metadata.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    spectra_dir.mkdir(parents=True, exist_ok=True)

    session = make_session()

    print(
        f"Searching WISeREP spectrum creation dates "
        f"{start.isoformat()} through {end.isoformat()} (inclusive)..."
    )

    try:
        candidates = collect_candidates(
            session,
            start,
            end,
            max_pages=args.max_pages,
            timeout=args.timeout,
            delay=args.delay,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(f"Search failed: {exc}", file=sys.stderr)
        return 1

    print(f"Found {len(candidates)} date-matching spectrum rows before SN filtering.")

    existing = read_existing_metadata(metadata_path)
    metadata = dict(existing)

    object_cache: dict[tuple[str, str], ObjectPage] = {}
    links_cache: dict[tuple[str, str], list[AsciiLink]] = {}

    saved = 0
    already_present = 0
    skipped_non_sn = 0
    skipped_missing_link = 0
    failed = 0

    for idx, candidate in enumerate(candidates, start=1):
        cache_key = (candidate.obj_id, candidate.iau)

        try:
            if cache_key not in object_cache:
                object_cache[cache_key] = fetch_object_page(
                    session,
                    candidate.iau,
                    candidate.obj_id,
                    timeout=args.timeout,
                    delay=args.delay,
                )
            obj = object_cache[cache_key]

            # Prefer a specific row classification when present; if the row is
            # generic "SN" or empty, use the deeper object-page classification.
            row_type = compact(candidate.row_type)
            if row_type and row_type.upper() not in {"SN", "UNKNOWN"}:
                sn_type = row_type
            else:
                sn_type = obj.sn_type

            if not is_supernova_type(sn_type):
                skipped_non_sn += 1
                if args.verbose:
                    print(
                        f"[{idx}/{len(candidates)}] skip non-SN "
                        f"{candidate.iau}: {sn_type}"
                    )
                continue

            row_z = normalize_redshift(candidate.row_redshift)
            redshift = row_z if row_z != "" else obj.redshift

            if cache_key not in links_cache:
                links_cache[cache_key] = extract_ascii_links(obj.html)
            link = resolve_ascii_link(candidate, links_cache[cache_key])

            if link is None:
                skipped_missing_link += 1
                print(
                    f"[{idx}/{len(candidates)}] WARNING: could not match ASCII file "
                    f"for IAU={candidate.iau} SpecID={candidate.spec_id!r} "
                    f"search_file={candidate.ascii_file!r}",
                    file=sys.stderr,
                )
                continue

            filename = stable_output_filename(
                candidate.iau,
                candidate.spec_id,
                link.url,
                link.filename,
            )
            destination = spectra_dir / filename

            if destination.exists() and not args.overwrite:
                already_present += 1
            else:
                download_ascii(
                    session,
                    link.url,
                    destination,
                    timeout=args.timeout,
                    delay=args.delay,
                )
                saved += 1

            # Only add metadata after the physical file exists.
            if destination.exists():
                metadata[filename] = {
                    "iau": candidate.iau,
                    "filename": filename,
                    "type": sn_type,
                    "redshift": redshift,
                }

            print(
                f"[{idx}/{len(candidates)}] {candidate.iau} "
                f"SpecID={candidate.spec_id or '?'} -> {filename}"
            )

        except Exception as exc:
            failed += 1
            print(
                f"[{idx}/{len(candidates)}] ERROR {candidate.iau} "
                f"SpecID={candidate.spec_id or '?'}: {exc}",
                file=sys.stderr,
            )

    write_metadata_atomic(metadata_path, metadata)

    print()
    print("Done.")
    print(f"  Newly downloaded:       {saved}")
    print(f"  Already present:        {already_present}")
    print(f"  Skipped non-SN:         {skipped_non_sn}")
    print(f"  Missing ASCII match:    {skipped_missing_link}")
    print(f"  Failed:                 {failed}")
    print(f"  Total metadata rows:    {len(metadata)}")
    print(f"  Spectra directory:      {spectra_dir}")
    print(f"  Metadata CSV:           {metadata_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())