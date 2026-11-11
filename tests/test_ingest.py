"""Unit tests for cleaning, HTML sectioning and chunking. No network or real files needed."""

from docuask.ingest import MIN_CHUNK_CHARS, Unit, chunk_units, clean_pdf_text, html_to_sections

RAW_PDF_PAGE = """
2

Federal Communications Commission   ·  Consumer and Governmental Affairs Bureau   ·  45 L Street NE
1-888-CALL-FCC (1-888-225-5322)   ·  TTY: 1-888-TELL-FCC (1-888-835-5322)

Filing a slamming complaint

The Communications Act and FCC rules hold telephone companies responsible for
the acts of their agents, including their telemarketers.

• File a complaint online at consumercomplaints.fcc.gov
• By phone: 1-888-CALL-FCC (1-888-225-5322); TTY: 1-888-TELL-FCC (1-888-835-
5322); ASL: 1-844-432-2275
• Providing a toll -free number that you can call.
Alternate formats

To request this article in an alternate format - braille, large print - send an
email to fcc504@fcc.gov.

Last Reviewed: 01/13/2022
"""


def test_clean_pdf_removes_letterhead_page_numbers_and_boilerplate():
    text = clean_pdf_text(RAW_PDF_PAGE)
    assert "45 L Street" not in text            # letterhead line
    assert not text.startswith("2")             # bare page number
    assert "fcc504" not in text                 # 'Alternate formats' paragraph
    assert "Last Reviewed" not in text


def test_clean_pdf_keeps_real_content_and_rejoins_wrapped_lines():
    text = clean_pdf_text(RAW_PDF_PAGE)
    assert "responsible for the acts of their agents" in text     # wrapped line re-joined
    assert "By phone: 1-888-CALL-FCC" in text                     # not mistaken for letterhead
    assert "1-888-835-5322" in text                               # number split across lines re-joined
    assert "toll-free" in text                                    # spacing artifact fixed
    assert text.count("•") == 3                              # bullets kept on their own lines
    assert "\nFiling a slamming complaint" in "\n" + text         # heading kept on its own line


HTML = """
<html><head><title>Test Guide | Federal Communications Commission</title></head><body>
<nav>Menu Home About</nav>
<article class="guide"><div>
  <style>.x{color:red}</style>
  <p>Intro paragraph that explains what this guide is about in a few sentences of plain text. """ + "Intro. " * 30 + """</p>
  <h3>Cramming</h3>
  <p>Cramming is the illegal act of placing unauthorized charges on your telephone bill. """ + "More detail. " * 20 + """</p>
  <h4>How does cramming occur?</h4>
  <ul><li>Third parties add charges.</li><li>Charges look generic.</li></ul>
  <h3>Tiny</h3><p>Short.</p>
  <h3>Know Your Rights</h3>
  <p>Your bill must be clear and identify the provider for each charge. """ + "Rights text. " * 20 + """</p>
  <h4>Printable Version</h4><p>Download the PDF</p>
  <p>Date Last Updated/Reviewed: Friday</p>
</div></article></body></html>
"""


def test_html_sections_split_on_headings_and_drop_site_chrome():
    sections = html_to_sections(HTML)
    headings = [h for h, _ in sections]
    joined = " ".join(b for _, b in sections)
    assert "Cramming" in headings
    assert "How does cramming occur?" in joined   # h4 stays inside its section
    assert "- Third parties add charges." in joined
    assert "Menu Home" not in joined              # <nav> removed
    assert "color:red" not in joined              # <style> removed
    assert "Download the PDF" not in joined       # footer after 'Printable Version' dropped


def test_html_short_sections_are_merged_not_lost():
    sections = dict(html_to_sections(HTML))
    assert "Tiny" not in sections                       # too short to be its own section
    assert any("Short." in body for body in sections.values())


def _units():
    sentence = "Customers can keep their phone number when they switch providers. "
    return [
        Unit("porting", "Porting", "u", "porting.pdf", 1, None, sentence * 30),
        Unit("porting", "Porting", "u", "porting.pdf", 2, None, "Page two text about porting fees. " * 5),
        Unit("bill", "Bill", "u", "bill.html", None, "Cramming", "Cramming\n" + "Unauthorized charges. " * 40),
    ]


def test_tiny_fragments_are_merged_not_kept_alone():
    unit = Unit("wea", "WEA", "u", "wea.pdf", 1, None,
                "Wireless Emergency Alerts (WEA)\n\n" + "Alerts are sent by authorized officials. " * 25)
    chunks = chunk_units([unit], chunk_size=300, chunk_overlap=60)
    assert all(len(c.text) >= MIN_CHUNK_CHARS for c in chunks)
    assert chunks[0].text.startswith("Wireless Emergency Alerts (WEA)")


def test_chunks_respect_size_and_keep_metadata():
    chunks = chunk_units(_units(), chunk_size=300, chunk_overlap=60)
    assert all(len(c.text) <= 300 + MIN_CHUNK_CHARS for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)          # ids unique
    page2 = [c for c in chunks if c.page == 2]
    assert page2 and all("Page two" in c.text for c in page2)        # never mixes pages
    assert page2[0].citation == "[porting.pdf, p. 2]"
    html_chunk = next(c for c in chunks if c.doc_id == "bill")
    assert html_chunk.citation == "[bill.html, § Cramming]"
    assert html_chunk.chunk_id.startswith("bill:s1:")


def test_consecutive_chunks_overlap():
    chunks = [c for c in chunk_units(_units(), 300, 60) if c.page == 1]
    assert len(chunks) > 2
    first, second = chunks[0].text, chunks[1].text
    assert second[:40] in first                                      # start of chunk 2 repeats end of chunk 1


def test_chunking_is_deterministic():
    a = chunk_units(_units(), 300, 60)
    b = chunk_units(_units(), 300, 60)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_html_footer_without_printable_heading_is_dropped():
    html = ("<html><head><title>G</title></head><body><article class='guide'><div>"
            "<p>" + "Real guide content about mobile speed tests. " * 10 + "</p>"
            "<p>Date Last Updated/Reviewed: Tuesday</p><p>Tags: Wireless Consumer Issues</p>"
            "<h3>FCC Consumer Resource Library</h3><p>Free printable tip cards.</p>"
            "</div></article></body></html>")
    joined = " ".join(b for _, b in html_to_sections(html))
    assert "Real guide content" in joined
    assert "Date Last Updated" not in joined and "Tags" not in joined and "tip cards" not in joined
