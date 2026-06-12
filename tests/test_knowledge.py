"""Knowledge lifecycle & version graph tests."""

import time
import pytest

from src.knowledge.lifecycle import (
    KnowledgeLifecycle, KnowledgeItem, FreshnessStatus,
)
from src.knowledge.version_graph import (
    KnowledgeGraph, KnowledgeNode, KnowledgeEdge, EdgeType,
)


class TestKnowledgeItem:
    """Test KnowledgeItem dataclass."""

    def test_defaults(self):
        item = KnowledgeItem(id="k1", content="test content")
        assert item.id == "k1"
        assert item.status == FreshnessStatus.FRESH
        assert item.ttl_days == 90
        assert item.importance == 0.5
        assert item.version == 1

    def test_age_days(self):
        item = KnowledgeItem(id="k1", content="x",
                           created_at=time.time() - 86400)
        assert item.age_days == pytest.approx(1.0, abs=0.1)

    def test_days_since_access(self):
        item = KnowledgeItem(id="k1", content="x",
                           last_accessed=time.time() - 86400)
        assert item.days_since_access == pytest.approx(1.0, abs=0.1)

    def test_custom_fields(self):
        item = KnowledgeItem(
            id="k2", content="important",
            source="agent-1", category="research",
            tags=["ai", "ml"], ttl_days=30,
            stale_grace_days=10, importance=0.9,
        )
        assert item.source == "agent-1"
        assert item.tags == ["ai", "ml"]
        assert item.ttl_days == 30


class TestKnowledgeLifecycle:
    """Test KnowledgeLifecycle manager."""

    @pytest.fixture
    def kl(self):
        return KnowledgeLifecycle()

    def test_add_and_get(self, kl):
        item = KnowledgeItem(id="k1", content="test", category="research")
        kl.add_item(item)
        assert kl.get("k1") is item
        assert kl.get("nonexistent") is None

    def test_add_indexes_category(self, kl):
        item = KnowledgeItem(id="k1", content="x", category="research")
        kl.add_item(item)
        assert "k1" in kl._by_category["research"]

    def test_add_indexes_tags(self, kl):
        item = KnowledgeItem(id="k1", content="x", tags=["ai", "ml"])
        kl.add_item(item)
        assert "k1" in kl._by_tag["ai"]
        assert "k1" in kl._by_tag["ml"]

    def test_add_indexes_source(self, kl):
        item = KnowledgeItem(id="k1", content="x", source="agent-1")
        kl.add_item(item)
        assert "k1" in kl._by_source["agent-1"]

    def test_update(self, kl):
        item = KnowledgeItem(id="k1", content="old")
        kl.add_item(item)
        assert kl.update("k1", content="new", importance=0.8)
        assert item.content == "new"
        assert item.importance == 0.8
        assert item.version == 2

    def test_update_nonexistent(self, kl):
        assert kl.update("nope", content="x") is False

    def test_remove(self, kl):
        item = KnowledgeItem(id="k1", content="x", category="research", tags=["ai"])
        kl.add_item(item)
        assert kl.remove("k1") is True
        assert kl.get("k1") is None
        assert "k1" not in kl._by_category.get("research", [])

    def test_remove_nonexistent(self, kl):
        assert kl.remove("nope") is False

    def test_access_updates_count_and_importance(self, kl):
        item = KnowledgeItem(id="k1", content="x", importance=0.5,
                           last_accessed=0, access_count=0)
        kl.add_item(item)
        kl.access("k1")
        assert item.access_count == 1
        assert item.importance > 0.5

    def test_access_archived_does_not_update_counters(self, kl):
        item = KnowledgeItem(id="k1", content="x",
                           status=FreshnessStatus.ARCHIVED,
                           access_count=0)
        kl.add_item(item)
        result = kl.access("k1")
        # Returns the item but does NOT update counters for archived items
        assert result is item
        assert item.access_count == 0

    def test_importance_capped_at_1(self, kl):
        item = KnowledgeItem(id="k1", content="x", importance=1.0)
        kl.add_item(item)
        kl.access("k1")
        assert item.importance == 1.0

    # ── Freshness ─────────────────────────────────────

    def test_check_freshness_fresh(self, kl):
        item = KnowledgeItem(id="k1", content="x", ttl_days=90)
        kl.add_item(item)
        results = kl.check_freshness()
        assert "k1" in results[FreshnessStatus.FRESH]

    def test_check_freshness_aging(self, kl):
        long_ago = time.time() - 80 * 86400  # 80 days ago (> 80% of 90)
        item = KnowledgeItem(id="k1", content="x", ttl_days=90,
                           created_at=long_ago)
        kl.add_item(item)
        results = kl.check_freshness()
        assert "k1" in results[FreshnessStatus.AGING]

    def test_check_freshness_stale(self, kl):
        long_ago = time.time() - 100 * 86400  # 100 days ago (> 90 ttl)
        item = KnowledgeItem(id="k1", content="x", ttl_days=90,
                           created_at=long_ago)
        kl.add_item(item)
        results = kl.check_freshness()
        assert "k1" in results[FreshnessStatus.STALE]

    def test_check_freshness_archived(self, kl):
        long_ago = time.time() - 150 * 86400  # > 90 + 30
        item = KnowledgeItem(id="k1", content="x", ttl_days=90,
                           stale_grace_days=30, created_at=long_ago)
        kl.add_item(item)
        results = kl.check_freshness()
        assert "k1" in results[FreshnessStatus.ARCHIVED]

    def test_check_freshness_reviewed_stays(self, kl):
        item = KnowledgeItem(id="k1", content="x",
                           status=FreshnessStatus.REVIEWED,
                           created_at=time.time() - 200 * 86400)
        kl.add_item(item)
        results = kl.check_freshness()
        assert "k1" in results[FreshnessStatus.REVIEWED]

    def test_mark_reviewed_resets_clock(self, kl):
        old_time = time.time() - 120 * 86400
        item = KnowledgeItem(id="k1", content="x", created_at=old_time,
                           ttl_days=30)
        kl.add_item(item)
        assert kl.mark_reviewed("k1") is True
        assert item.status == FreshnessStatus.REVIEWED
        assert item.created_at > old_time + 115 * 86400  # near now
        assert item.ttl_days >= 90

    def test_mark_reviewed_nonexistent(self, kl):
        assert kl.mark_reviewed("nope") is False

    # ── Query ─────────────────────────────────────────

    def test_get_fresh_by_category(self, kl):
        kl.add_item(KnowledgeItem(id="k1", content="x", category="research",
                                  importance=0.8))
        kl.add_item(KnowledgeItem(id="k2", content="y", category="engineering",
                                  importance=0.3))
        results = kl.get_fresh(category="research")
        assert len(results) == 1
        assert results[0].id == "k1"

    def test_get_fresh_respects_importance(self, kl):
        kl.add_item(KnowledgeItem(id="k1", content="x", importance=0.2))
        kl.add_item(KnowledgeItem(id="k2", content="y", importance=0.9))
        results = kl.get_fresh(min_importance=0.5)
        assert len(results) == 1
        assert results[0].id == "k2"

    def test_get_fresh_skips_archived(self, kl):
        kl.add_item(KnowledgeItem(id="k1", content="x",
                                  status=FreshnessStatus.ARCHIVED))
        results = kl.get_fresh()
        assert len(results) == 0

    def test_get_stale(self, kl):
        long_ago = time.time() - 100 * 86400
        kl.add_item(KnowledgeItem(id="k1", content="x", ttl_days=90,
                                  created_at=long_ago))
        kl.check_freshness()
        stale = kl.get_stale()
        assert len(stale) == 1
        assert stale[0].id == "k1"

    def test_search_by_content(self, kl):
        kl.add_item(KnowledgeItem(id="k1", content="Python is great for ML"))
        kl.add_item(KnowledgeItem(id="k2", content="Rust is fast for systems"))
        results = kl.search("Python")
        assert len(results) == 1
        assert results[0].id == "k1"

    def test_search_by_tag(self, kl):
        kl.add_item(KnowledgeItem(id="k1", content="x", tags=["machine-learning"]))
        kl.add_item(KnowledgeItem(id="k2", content="y", tags=["web-dev"]))
        results = kl.search("machine-learning")
        assert len(results) >= 1
        assert results[0].id == "k1"

    def test_search_skips_archived(self, kl):
        kl.add_item(KnowledgeItem(id="k1", content="Python stuff",
                                  status=FreshnessStatus.ARCHIVED))
        results = kl.search("Python")
        assert len(results) == 0

    def test_get_stats(self, kl):
        kl.add_item(KnowledgeItem(id="k1", content="x", category="research"))
        kl.add_item(KnowledgeItem(id="k2", content="y", category="engineering"))
        stats = kl.get_stats()
        assert stats["total_items"] == 2
        assert stats["by_category"]["research"] == 1

    def test_archive_stale(self, kl):
        long_ago = time.time() - 130 * 86400  # past 90+30
        item = KnowledgeItem(id="k1", content="x", ttl_days=90,
                           stale_grace_days=30, created_at=long_ago)
        kl.add_item(item)
        kl.check_freshness()
        assert item.status == FreshnessStatus.STALE or item.status == FreshnessStatus.ARCHIVED


class TestKnowledgeGraph:
    """Test KnowledgeGraph."""

    @pytest.fixture
    def kg(self):
        return KnowledgeGraph()

    def test_add_and_get_node(self, kg):
        node = kg.add_node(KnowledgeNode(title="Python", content="A language"))
        retrieved = kg.get_node(node.id)
        assert retrieved is node
        assert retrieved.title == "Python"

    def test_get_nonexistent_node(self, kg):
        assert kg.get_node("nope") is None

    def test_add_edge(self, kg):
        n1 = kg.add_node(KnowledgeNode(title="A"))
        n2 = kg.add_node(KnowledgeNode(title="B"))
        edge = KnowledgeEdge(n1.id, n2.id, EdgeType.SUPPORTS)
        kg.add_edge(edge)
        assert len(kg._out_edges[n1.id]) == 1
        assert len(kg._in_edges[n2.id]) == 1

    def test_remove_edge(self, kg):
        n1 = kg.add_node(KnowledgeNode(title="A"))
        n2 = kg.add_node(KnowledgeNode(title="B"))
        kg.add_edge(KnowledgeEdge(n1.id, n2.id, EdgeType.REFERENCES))
        count = kg.remove_edge(n1.id, n2.id, EdgeType.REFERENCES)
        assert count == 2  # removes from both out_edges and in_edges

    def test_update_node_creates_new_version(self, kg):
        n1 = kg.add_node(KnowledgeNode(title="Python", content="v1", version=1))
        n2 = kg.update_node(n1.id, "v2")
        assert n2.id != n1.id
        assert n2.version == 2
        assert n2.content == "v2"
        # Supersedes edge created
        edges = kg._out_edges[n2.id]
        assert any(e.type == EdgeType.SUPERSEDES for e in edges)

    def test_update_node_in_place(self, kg):
        n1 = kg.add_node(KnowledgeNode(title="X", content="v1", version=1))
        n2 = kg.update_node(n1.id, "v2", create_new_version=False)
        assert n2.id == n1.id
        assert n2.content == "v2"
        assert n2.version == 2

    def test_update_nonexistent_node_raises(self, kg):
        with pytest.raises(KeyError):
            kg.update_node("nope", "content")

    # ── Queries ───────────────────────────────────────

    def test_get_related_bfs(self, kg):
        n1 = kg.add_node(KnowledgeNode(title="A"))
        n2 = kg.add_node(KnowledgeNode(title="B"))
        n3 = kg.add_node(KnowledgeNode(title="C"))
        kg.add_edge(KnowledgeEdge(n1.id, n2.id, EdgeType.RELATED))
        kg.add_edge(KnowledgeEdge(n2.id, n3.id, EdgeType.DERIVES_FROM))

        related = kg.get_related(n1.id, depth=2)
        ids = [r[0].id for r in related]
        assert n2.id in ids
        assert n3.id in ids

    def test_get_related_depth_1(self, kg):
        n1 = kg.add_node(KnowledgeNode(title="A"))
        n2 = kg.add_node(KnowledgeNode(title="B"))
        n3 = kg.add_node(KnowledgeNode(title="C"))
        kg.add_edge(KnowledgeEdge(n1.id, n2.id, EdgeType.RELATED))
        kg.add_edge(KnowledgeEdge(n2.id, n3.id, EdgeType.DERIVES_FROM))

        related = kg.get_related(n1.id, depth=1)
        assert len(related) == 1
        assert related[0][0].id == n2.id

    def test_get_related_nonexistent(self, kg):
        assert kg.get_related("nope") == []

    def test_get_related_filter_by_type(self, kg):
        n1 = kg.add_node(KnowledgeNode(title="A"))
        n2 = kg.add_node(KnowledgeNode(title="B"))
        kg.add_edge(KnowledgeEdge(n1.id, n2.id, EdgeType.CONTRADICTS))

        related = kg.get_related(n1.id, edge_types=[EdgeType.CONTRADICTS])
        assert len(related) == 1

        related2 = kg.get_related(n1.id, edge_types=[EdgeType.SUPPORTS])
        assert len(related2) == 0

    def test_find_contradictions(self, kg):
        n1 = kg.add_node(KnowledgeNode(title="Claim A"))
        n2 = kg.add_node(KnowledgeNode(title="Claim B"))
        kg.add_edge(KnowledgeEdge(n1.id, n2.id, EdgeType.CONTRADICTS))
        kg.add_edge(KnowledgeEdge(n1.id, n2.id, EdgeType.SUPPORTS))

        contradictions = kg.find_contradictions()
        assert len(contradictions) == 1

    def test_find_contradictions_empty(self, kg):
        n1 = kg.add_node(KnowledgeNode(title="A"))
        n2 = kg.add_node(KnowledgeNode(title="B"))
        kg.add_edge(KnowledgeEdge(n1.id, n2.id, EdgeType.SUPPORTS))
        assert len(kg.find_contradictions()) == 0

    # ── Version chain ─────────────────────────────────

    def test_get_version_chain(self, kg):
        v1 = kg.add_node(KnowledgeNode(title="Doc", content="v1", version=1))
        v2 = kg.add_node(KnowledgeNode(title="Doc", content="v2", version=2))
        v3 = kg.add_node(KnowledgeNode(title="Doc", content="v3", version=3))
        kg.add_edge(KnowledgeEdge(v2.id, v1.id, EdgeType.SUPERSEDES))
        kg.add_edge(KnowledgeEdge(v3.id, v2.id, EdgeType.SUPERSEDES))

        chain = kg.get_version_chain(v2.id)
        assert len(chain) == 3
        assert chain[0].id == v1.id
        assert chain[1].id == v2.id
        assert chain[2].id == v3.id

    def test_get_version_chain_single(self, kg):
        v1 = kg.add_node(KnowledgeNode(title="Only"))
        chain = kg.get_version_chain(v1.id)
        assert len(chain) == 1

    def test_get_version_chain_nonexistent(self, kg):
        assert kg.get_version_chain("nope") == []

    # ── Stats ─────────────────────────────────────────

    def test_get_stats(self, kg):
        n1 = kg.add_node(KnowledgeNode(title="A"))
        n2 = kg.add_node(KnowledgeNode(title="B"))
        kg.add_edge(KnowledgeEdge(n1.id, n2.id, EdgeType.SUPPORTS))
        kg.add_edge(KnowledgeEdge(n2.id, n1.id, EdgeType.CONTRADICTS))

        stats = kg.get_stats()
        assert stats["nodes"] == 2
        assert stats["edges"] == 2
        assert stats["contradictions"] == 1
        assert "supports" in stats["by_edge_type"]


class TestEdgeType:
    """Test EdgeType enum."""

    def test_edge_types(self):
        assert EdgeType.DERIVES_FROM.value == "derives_from"
        assert EdgeType.CONTRADICTS.value == "contradicts"
        assert EdgeType.SUPERSEDES.value == "supersedes"
        assert EdgeType.SUPPORTS.value == "supports"
        assert len(list(EdgeType)) == 6


class TestFreshnessStatus:
    """Test FreshnessStatus enum."""

    def test_statuses(self):
        assert FreshnessStatus.FRESH.value == "fresh"
        assert FreshnessStatus.AGING.value == "aging"
        assert FreshnessStatus.STALE.value == "stale"
        assert FreshnessStatus.ARCHIVED.value == "archived"
        assert FreshnessStatus.REVIEWED.value == "reviewed"
