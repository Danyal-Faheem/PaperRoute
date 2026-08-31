from urllib.parse import unquote_plus

import pytest

from paperroute.arxiv_client import ArxivClient

ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/2401.12345v2</id><title>duplicate</title><summary>old</summary></entry>
<entry><id>http://arxiv.org/abs/2401.12345v2</id><title>  A\n paper </title>
<summary> An abstract\nwith spaces. </summary><author><name>Ada Lovelace</name></author>
<category term="cs.AI"/><link type="text/html" href="https://arxiv.org/abs/2401.12345"/>
<link type="application/pdf" href="https://arxiv.org/pdf/2401.12345"/></entry>
<entry><id>http://arxiv.org/abs/2402.00001</id><title>Second</title><summary>Two</summary></entry></feed>"""


def test_parse_feed_normalizes_text_and_deduplicates_ids():
    papers = ArxivClient._parse_feed(ATOM)
    assert [paper.arxiv_id for paper in papers] == ["2401.12345v2", "2402.00001"]
    assert papers[0].title == "A paper"
    assert papers[0].authors == ["Ada Lovelace"]


class FakeResponse:
    text = ATOM
    content = b"pdf"

    def raise_for_status(self):
        return None


class FakeTransport:
    def __init__(self):
        self.urls = []

    async def get(self, url, **kwargs):
        self.urls.append(url)
        return FakeResponse()


@pytest.mark.asyncio
async def test_search_limits_results_and_adds_categories(tmp_path):
    transport = FakeTransport()
    client = ArxivClient(cache_dir=tmp_path, transport=transport)
    papers = await client.search("graph neural networks", max_results=99, categories=["cs.LG"])
    assert len(papers) == 2
    assert "max_results=20" in transport.urls[0]
    assert "cat%3Acs.LG" in transport.urls[0]


@pytest.mark.asyncio
async def test_search_preserves_fielded_query_and_groups_topic_before_categories(tmp_path):
    transport = FakeTransport()
    client = ArxivClient(cache_dir=tmp_path, transport=transport)
    query = 'all:"NVIDIA binaries" AND all:"AMD hardware" AND cat:cs.AI OR cat:cs.CL'

    await client.search(query, categories=["cs.AI", "cs.CL"])

    search_query = unquote_plus(transport.urls[0].split("search_query=", 1)[1].split("&", 1)[0])
    assert "all:all:" not in search_query
    assert search_query == f"({query}) AND (cat:cs.AI OR cat:cs.CL)"
