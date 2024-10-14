# tests/test_tools.py

import unittest
from unittest.mock import patch, MagicMock
from graph_reactagent.tools import search, power, get_webex_user_info  # type: ignore


class TestSearch(unittest.TestCase):
    @patch("graph_reactagent.tools.TavilySearchResults")
    def test_search_success(self, mock_tavily):
        mock_tavily_instance = MagicMock()
        mock_tavily_instance.invoke.return_value = [
            {"title": "Test Result 1", "url": "http://test1.com"},
            {"title": "Test Result 2", "url": "http://test2.com"},
        ]
        mock_tavily.return_value = mock_tavily_instance

        config = MagicMock()
        config.get.return_value = {"max_results": 2}

        result = search("test query", config=config)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["title"], "Test Result 1")
        self.assertEqual(result[1]["url"], "http://test2.com")
        mock_tavily_instance.invoke.assert_called_once_with({"query": "test query"})

    @patch("graph_reactagent.tools.TavilySearchResults")
    def test_search_no_results(self, mock_tavily):
        mock_tavily_instance = MagicMock()
        mock_tavily_instance.invoke.return_value = []
        mock_tavily.return_value = mock_tavily_instance

        config = MagicMock()
        config.get.return_value = {"max_results": 5}

        result = search("no results query", config=config)

        self.assertEqual(result, [])

    def test_search_invalid_config(self):
        config = MagicMock()
        config.get.return_value = {}  # Empty configurable dict

        result = search("test query", config=config)

        self.assertIsNotNone(result)  # The function should not crash
        # Additional assertions can be added based on how you want to handle this case


class TestPower(unittest.TestCase):
    def test_power_positive_numbers(self):
        self.assertEqual(power(2, 3), 8)
        self.assertEqual(power(5, 2), 25)

    def test_power_zero_exponent(self):
        self.assertEqual(power(5, 0), 1)
        self.assertEqual(power(0, 0), 1)

    def test_power_negative_exponent(self):
        self.assertEqual(power(2, -2), 0.25)

    def test_power_zero_base(self):
        self.assertEqual(power(0, 5), 0)

    def test_power_large_numbers(self):
        self.assertEqual(power(10, 10), 10000000000)


class TestGetWebexUserInfo(unittest.TestCase):
    def test_get_webex_user_info_success(self):
        config = MagicMock()
        config.get.return_value = {
            "displayName": "John Doe",
            "email": "john@example.com",
        }

        result = get_webex_user_info(config)

        self.assertEqual(result["displayName"], "John Doe")
        self.assertEqual(result["email"], "john@example.com")

    def test_get_webex_user_info_empty(self):
        config = MagicMock()
        config.get.return_value = {}

        result = get_webex_user_info(config)

        self.assertEqual(result["displayName"], "")
        self.assertEqual(result["email"], "")

    def test_get_webex_user_info_partial(self):
        config = MagicMock()
        config.get.return_value = {"displayName": "Jane Doe"}

        result = get_webex_user_info(config)

        self.assertEqual(result["displayName"], "Jane Doe")
        self.assertEqual(result["email"], "")


if __name__ == "__main__":
    unittest.main()
