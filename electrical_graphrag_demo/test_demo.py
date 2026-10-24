"""Offline unit tests. These do not claim to validate live Neo4j or Ollama."""
import json
import unittest
from unittest.mock import patch

import demo


class RetrievalTests(unittest.TestCase):
    def test_baseline_context_cannot_contain_unretrieved_panel_report(self):
        context = demo.build_context([demo.DOCS[2]])
        self.assertIn("[D3]", context)
        self.assertNotIn("[D1]", context)
        self.assertNotIn("panel failed", context)

    def test_context_includes_relationship_direction_and_sources(self):
        path = {"names": ["Electrical panel issue", "Wiring completion", "Final inspection"],
                "relationTypes": ["BLOCKS", "REQUIRED_FOR"], "sourceIds": ["D2", "D3"]}
        context = demo.build_context(demo.DOCS, [path])
        self.assertIn("[BLOCKS; D2]", context)
        self.assertIn("[REQUIRED_FOR; D3]", context)
        self.assertIn("[D1]", context)

    def test_expansion_uses_retrieved_seed_ids_not_hardcoded_question_answer(self):
        path = {"entityIds": ["panel_issue", "wiring", "final_inspection"],
                "names": ["Electrical panel issue", "Wiring completion", "Final inspection"],
                "relationTypes": ["BLOCKS", "REQUIRED_FOR"], "sourceIds": ["D2", "D3"]}
        with patch.object(demo, "query", side_effect=[[path], demo.DOCS]) as call:
            docs, paths = demo.graph_expand(object(), [demo.DOCS[2]])
        self.assertEqual(call.call_args_list[0].kwargs["seedIds"], ["D3"])
        self.assertEqual(call.call_args_list[1].kwargs["sourceIds"], ["D2", "D3"])
        self.assertEqual(len(docs), 3)
        self.assertEqual(paths, [path])

    def test_full_vector_context_already_contains_all_three_facts(self):
        context = demo.build_context(demo.DOCS)
        for doc in demo.DOCS:
            self.assertIn(doc["text"], context)

    def test_same_prompt_for_both_modes_and_nonstreaming_response(self):
        with patch.object(demo, "ollama_post", return_value={"message": {"content": "answer"}}) as call:
            self.assertEqual(demo.generate("question", "evidence"), "answer")
        payload = call.call_args.args[1]
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][0]["content"], demo.SYSTEM_PROMPT)

    def test_inconsistent_embedding_dimensions_rejected(self):
        with patch.object(demo, "ollama_post", return_value={"embeddings": [[1, 0], [1]]}):
            with self.assertRaises(ValueError):
                demo.embed(["a", "b"])

    def test_fixture_relationships_have_real_sources(self):
        ids = {d["id"] for d in demo.DOCS}
        for edge in demo.EDGES:
            self.assertIn(edge["source"], ids)


if __name__ == "__main__":
    unittest.main()
