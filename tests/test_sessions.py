import concurrent.futures
import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path

from demo.sessions import (
    MAX_JSON_STRING_CHARACTERS,
    MAX_MESSAGE_CHARACTERS,
    MAX_TITLE_CHARACTERS,
    SessionStore,
)


class SessionStoreTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.database_path = self.root / "state" / "sessions.sqlite3"
        self.store = SessionStore(self.database_path)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_sessions_persist_and_use_canonical_uuid_ids(self):
        created = self.store.create_session("Qwen/Qwen3-VL-4B", "Charts")
        self.assertEqual(str(uuid.UUID(created["id"])), created["id"])
        self.assertEqual(created["messages"], [])
        self.assertEqual(created["media"], [])

        reopened = SessionStore(self.database_path)
        loaded = reopened.get_session(created["id"])

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["title"], "Charts")
        self.assertEqual(loaded["model_id"], "Qwen/Qwen3-VL-4B")
        self.assertEqual(reopened.list_sessions()[0]["id"], created["id"])

    def test_unsupported_schema_version_is_rejected(self):
        incompatible_path = self.root / "incompatible.sqlite3"
        with sqlite3.connect(incompatible_path) as connection:
            connection.execute("PRAGMA user_version = 2")

        with self.assertRaises(RuntimeError):
            SessionStore(incompatible_path)

    def test_messages_preserve_reasoning_metrics_and_order(self):
        session = self.store.create_session()
        first = self.store.append_message(session["id"], "user", "Read this chart")
        second = self.store.append_message(
            session["id"],
            "assistant",
            "Revenue rises.",
            reasoning="The bars increase from left to right.",
            metrics={
                "tokens_per_second": 31.5,
                "generated_tokens": 18,
                "finish_reason": "eos",
            },
        )

        loaded = self.store.get_session(session["id"])

        self.assertEqual(str(uuid.UUID(first["id"])), first["id"])
        self.assertEqual(str(uuid.UUID(second["id"])), second["id"])
        self.assertEqual([item["position"] for item in loaded["messages"]], [0, 1])
        self.assertEqual(loaded["messages"][1]["reasoning"], second["reasoning"])
        self.assertEqual(loaded["messages"][1]["metrics"], second["metrics"])
        self.assertEqual(loaded["title"], "Read this chart")

    def test_append_turn_is_atomic_and_preserves_assistant_metadata(self):
        session = self.store.create_session()
        user, assistant = self.store.append_turn(
            session["id"],
            "Read the formula",
            "E=mc^2",
            reasoning="Recognized the symbols",
            metrics={"tokens": 7},
        )
        self.assertEqual((user["position"], assistant["position"]), (0, 1))
        loaded = self.store.get_session(session["id"])
        self.assertEqual(
            [message["role"] for message in loaded["messages"]],
            ["user", "assistant"],
        )
        self.assertEqual(loaded["messages"][1]["reasoning"], "Recognized the symbols")
        self.assertEqual(loaded["messages"][1]["metrics"], {"tokens": 7})
        with self.assertRaises(ValueError):
            self.store.append_turn(
                session["id"],
                "Second turn",
                "answer",
                metrics={"bad": float("nan")},
            )
        self.assertEqual(len(self.store.get_session(session["id"])["messages"]), 2)

    def test_rename_and_list_sort_by_most_recent_change(self):
        first = self.store.create_session(title="First")
        second = self.store.create_session(title="Second")

        self.assertTrue(self.store.rename_session(first["id"], "Renamed"))
        self.assertFalse(self.store.rename_session(str(uuid.uuid4()), "Missing"))

        sessions = self.store.list_sessions()
        self.assertEqual(sessions[0]["id"], first["id"])
        self.assertEqual(sessions[0]["title"], "Renamed")
        self.assertEqual({item["id"] for item in sessions}, {first["id"], second["id"]})

    def test_media_is_publicly_sanitized_and_privately_retrievable(self):
        session = self.store.create_session()
        message = self.store.append_message(session["id"], "user", "Describe")
        stored_path = self.root / "media" / "chart.png"
        media = self.store.register_media(
            session["id"],
            message_id=message["id"],
            stored_path=stored_path,
            media_type="image",
            original_name="chart.png",
            mime_type="image/png",
            size_bytes=321,
            sha256="a" * 64,
            metadata={"width": 640, "height": 480},
        )

        self.assertNotIn("stored_path", media)
        self.assertNotIn("stored_path", self.store.list_media(session["id"])[0])
        self.assertNotIn("stored_path", self.store.get_media(media["id"]))
        self.assertNotIn(
            "stored_path", self.store.get_session(session["id"])["media"][0]
        )
        private = self.store.get_media(media["id"], include_stored_path=True)
        self.assertEqual(private["stored_path"], stored_path)
        self.assertEqual(private["metadata"], {"height": 480, "width": 640})

    def test_reset_clears_conversation_and_returns_media_paths(self):
        session = self.store.create_session(title="Before reset")
        message = self.store.append_message(session["id"], "user", "Prompt")
        media_path = self.root / "media" / "scan.png"
        self.store.register_media(
            session["id"],
            message_id=message["id"],
            stored_path=media_path,
            media_type="image",
            original_name="scan.png",
            mime_type="image/png",
            size_bytes=10,
        )

        paths = self.store.reset_conversation(session["id"])
        loaded = self.store.get_session(session["id"])

        self.assertEqual(paths, [media_path])
        self.assertEqual(loaded["title"], "New chat")
        self.assertEqual(loaded["messages"], [])
        self.assertEqual(loaded["media"], [])
        self.assertEqual(self.store.reset_conversation(str(uuid.uuid4())), None)

        alias_session = self.store.create_session()
        self.assertEqual(self.store.reset_session(alias_session["id"]), [])

    def test_delete_cascades_and_returns_media_paths(self):
        session = self.store.create_session()
        message = self.store.append_message(session["id"], "user", "Prompt")
        paths = [self.root / "media" / "a.png", self.root / "media" / "b.png"]
        for path in paths:
            self.store.register_media(
                session["id"],
                message_id=message["id"],
                stored_path=path,
                media_type="image",
                original_name=path.name,
                mime_type="image/png",
                size_bytes=10,
            )

        returned = self.store.delete_session(session["id"])

        self.assertEqual(returned, paths)
        self.assertIsNone(self.store.get_session(session["id"]))
        self.assertEqual(self.store.delete_session(session["id"]), None)
        with sqlite3.connect(self.database_path) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 0
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM media").fetchone()[0], 0
            )

    def test_cross_session_media_message_is_rejected(self):
        first = self.store.create_session()
        second = self.store.create_session()
        message = self.store.append_message(first["id"], "user", "Prompt")

        with self.assertRaises(ValueError):
            self.store.register_media(
                second["id"],
                message_id=message["id"],
                stored_path=self.root / "bad.png",
                media_type="image",
                original_name="bad.png",
                mime_type="image/png",
                size_bytes=10,
            )

    def test_validation_rejects_unbounded_and_malformed_values(self):
        session = self.store.create_session()
        cases = [
            lambda: self.store.create_session(title="x" * (MAX_TITLE_CHARACTERS + 1)),
            lambda: self.store.list_sessions(0),
            lambda: self.store.get_session("not-a-uuid"),
            lambda: self.store.append_message(session["id"], "invalid", "text"),
            lambda: self.store.append_message(
                session["id"], "user", "x" * (MAX_MESSAGE_CHARACTERS + 1)
            ),
            lambda: self.store.append_message(session["id"], "assistant", ""),
            lambda: self.store.append_message(
                session["id"], "assistant", "ok", metrics={"loss": float("nan")}
            ),
            lambda: self.store.append_message(
                session["id"],
                "assistant",
                "ok",
                metrics={"text": "x" * (MAX_JSON_STRING_CHARACTERS + 1)},
            ),
            lambda: self.store.register_media(
                session["id"],
                stored_path="relative.png",
                media_type="image",
                original_name="relative.png",
                mime_type="image/png",
                size_bytes=10,
            ),
            lambda: self.store.register_media(
                session["id"],
                stored_path=self.root / "bad.png",
                media_type="image",
                original_name="../bad.png",
                mime_type="image/png",
                size_bytes=10,
            ),
            lambda: self.store.register_media(
                session["id"],
                stored_path=self.root / "bad.mp4",
                media_type="video",
                original_name="bad.mp4",
                mime_type="image/png",
                size_bytes=10,
            ),
            lambda: self.store.register_media(
                session["id"],
                stored_path=self.root / "bad.png",
                media_type="image",
                original_name="bad.png",
                mime_type="image/png",
                size_bytes=True,
            ),
        ]

        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises((TypeError, ValueError)):
                    case()

    def test_unknown_parent_ids_are_rejected(self):
        missing_session = str(uuid.uuid4())

        with self.assertRaises(KeyError):
            self.store.append_message(missing_session, "user", "Prompt")
        with self.assertRaises(KeyError):
            self.store.register_media(
                missing_session,
                stored_path=self.root / "missing.png",
                media_type="image",
                original_name="missing.png",
                mime_type="image/png",
                size_bytes=10,
            )

    def test_concurrent_appends_are_serialized(self):
        session = self.store.create_session(title="Concurrent")

        def append(index):
            return self.store.append_message(session["id"], "user", f"message {index}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            messages = list(executor.map(append, range(40)))

        loaded = self.store.get_session(session["id"])
        self.assertEqual(len(messages), 40)
        self.assertEqual(len({message["id"] for message in messages}), 40)
        self.assertEqual(
            [message["position"] for message in loaded["messages"]], list(range(40))
        )


if __name__ == "__main__":
    unittest.main()
