from ha.data.hiaq_scrape import extract_links, fix_url, in_scope, normalise_url

FIXES = [['^http://', 'https://'], ['/iec/foportal/iec/foportal/', '/iec/foportal/']]


def test_sitemap_url_fixes():
    raw = "http://www.incometax.gov.in/iec/foportal/iec/foportal/help/company/x/#taxslabs"
    assert fix_url(raw, FIXES) == "https://www.incometax.gov.in/iec/foportal/help/company/x"

ALLOW = [r'^https://www\.incometax\.gov\.in/iec/foportal/help(/[^?#]*)?$']
DENY = ['/iec/foportal/(hi|te|kn)/', r'\.(pdf|zip)$']


def test_scope_rules():
    assert in_scope("https://www.incometax.gov.in/iec/foportal/help/all-topics/x", ALLOW, DENY)
    assert in_scope("https://www.incometax.gov.in/iec/foportal/help", ALLOW, DENY)
    assert not in_scope("https://www.incometax.gov.in/iec/foportal/hi/help", ALLOW, DENY)
    assert not in_scope("https://www.incometax.gov.in/iec/foportal/help/a.pdf", ALLOW, DENY)
    assert not in_scope("https://eportal.incometax.gov.in/iec/foportal/help", ALLOW, DENY)
    assert not in_scope("https://www.incometax.gov.in/iec/foportal/help?q=1", ALLOW, DENY)


def test_links_resolved_and_fragments_dropped():
    html = ('<a href="/iec/foportal/help/x#top">x</a><a href="mailto:a@b.c">m</a>'
            '<a href="y/">y</a><a href="javascript:void(0)">j</a>')
    links = extract_links(html, "https://www.incometax.gov.in/iec/foportal/help/")
    assert links == ["https://www.incometax.gov.in/iec/foportal/help/x",
                     "https://www.incometax.gov.in/iec/foportal/help/y"]
    assert normalise_url("https://a.b/c/#frag") == "https://a.b/c"
    assert normalise_url("https://a.b/") == "https://a.b/"
