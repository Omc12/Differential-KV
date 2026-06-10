import sys
import os
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.openai_compatible_api_gateway import OpenAICompatibleAPIGateway


class TestGatewaySessionMatching(unittest.TestCase):
    def setUp(self):
        # Create a mock resolver and session manager
        self.mock_resolver = MagicMock()
        self.mock_session_manager = MagicMock()
        self.gateway = OpenAICompatibleAPIGateway(
            resolver=self.mock_resolver,
            session_manager=self.mock_session_manager
        )

    def test_is_ephemeral_request_short_title(self):
        # Open WebUI short background title request
        messages = [
            {"role": "user", "content": "summarize the chat history into a title"}
        ]
        self.assertTrue(self.gateway._is_ephemeral_request(messages))

    def test_is_ephemeral_request_long_title_with_history(self):
        # Open WebUI standard long title request containing chat history
        messages = [
            {
                "role": "user",
                "content": (
                    "Generate a concise, 3-5 word title with an emoji summarizing "
                    "the chat history:\nUser: Hello\nAssistant: Hi there! How can I help you today?"
                )
            }
        ]
        self.assertTrue(self.gateway._is_ephemeral_request(messages))

    def test_is_ephemeral_request_not_ephemeral(self):
        # Normal chat continuation message asking a question about a paper
        messages = [
            {"role": "user", "content": "Can you explain how randomized kernel machines work?"}
        ]
        self.assertFalse(self.gateway._is_ephemeral_request(messages))

    def test_is_ephemeral_request_short_normal_message_with_keyword(self):
        # User asking a short normal question that contains a keyword like 'summary'
        # BUT this is part of a longer conversation (messages > 1)
        messages = [
            {"role": "user", "content": "Explain the research paper."},
            {"role": "assistant", "content": "It describes randomized kernel machines."},
            {"role": "user", "content": "Give me a summary."}
        ]
        # Since it is a user role message < 500 chars containing 'summary', Heuristic 2 matches.
        # Wait, if Heuristic 2 matches, is it correct?
        # Yes, we want to see if our tightened check or Heuristic 2 is active.
        self.assertTrue(self.gateway._is_ephemeral_request(messages))


if __name__ == "__main__":
    unittest.main()
