"""Offline tests; no SDK, model download, credentials, or live services."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hybrid_search as app


class Model:
    def encode(self, question, normalize_embeddings):
        assert normalize_embeddings
        assert question.startswith(app.QUERY_PREFIX)
        return SimpleNamespace(tolist=lambda: [0.0] * 1024)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.chunks = app.read_chunks(app.ROOT / 'data' / 'security_policy.txt')

    def test_document_and_fact_provenance(self):
        self.assertEqual(set(self.chunks), {8, 24, 51, 79})
        for _, _, _, chunk_id in app.FACTS:
            self.assertIn(chunk_id, self.chunks)
        self.assertEqual(len(app.FACTS), 6)

    def test_linker_handles_aliases_without_substring_matches(self):
        self.assertIn('external contractor', app.link_entities('External contractors'))
        self.assertNotIn('external contractor', app.link_entities('subcontractor'))
        self.assertGreaterEqual(app.link_entities(app.QUESTION),
                                {'external contractor', 'customer data'})

    def test_merges_vector_and_graph_evidence(self):
        captured = {}

        def vector_query(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(matches=[SimpleNamespace(
                chunk_id=51, text=self.chunks[51], score=0.8)])

        def graph_query(driver, cypher, **params):
            self.assertEqual(params['doc'], app.DOC_ID)
            if cypher == app.TRAVERSE:
                self.assertIn('customer data', params['seeds'])
                return [{'source': a, 'relation': r, 'target': b, 'chunk_id': c}
                        for a, r, b, c in app.FACTS]
            return [{'chunk_id': c, 'text': self.chunks[c]} for c in params['ids']]

        with patch.object(app, 'query', side_effect=graph_query):
            result = app.search(SimpleNamespace(documents=SimpleNamespace(search=vector_query)), None,
                                Model(), app.QUESTION, 1)
        self.assertEqual(captured['namespace'], app.DOC_ID)
        self.assertEqual(len(captured['score_by'][0]['values']), 1024)
        self.assertEqual(captured['score_by'][0]['field'], 'embedding')
        self.assertEqual(captured['filter']['document_id'], {'$eq': app.DOC_ID})
        self.assertEqual(captured['filter']['embedding_model'], {'$eq': app.MODEL})
        self.assertEqual(result['graph_added_chunk_ids'], [8, 24, 79])
        self.assertEqual([c['chunk_id'] for c in result['context']], [8, 24, 51, 79])

    def test_empty_vector_results_require_ingestion(self):
        index = SimpleNamespace(documents=SimpleNamespace(search=lambda **kwargs: SimpleNamespace(matches=[])))
        with self.assertRaisesRegex(ValueError, 'Run ingest first'):
            app.search(index, None, Model(), app.QUESTION, 1)

    def test_missing_source_chunk_is_reported(self):
        index = SimpleNamespace(documents=SimpleNamespace(search=lambda **kwargs: SimpleNamespace(matches=[
            SimpleNamespace(chunk_id=51, text=self.chunks[51], score=0.8)])))
        with patch.object(app, 'query', return_value=[]):
            with self.assertRaisesRegex(ValueError, 'Missing Neo4j source chunks'):
                app.search(index, None, Model(), app.QUESTION, 1)


if __name__ == '__main__':
    unittest.main()
