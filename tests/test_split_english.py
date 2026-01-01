import unittest

from dp.align_train import split_english


class TestSplitEnglish(unittest.TestCase):
    def test_abbreviation(self) -> None:
        text = "We met Dr. Smith. He said e.g. examples."
        parts = split_english(text, min_len=1)
        self.assertEqual(parts, ["We met Dr. Smith.", "He said e.g. examples."])

    def test_decimal(self) -> None:
        text = "The value is 1.5 kg. Done."
        parts = split_english(text, min_len=1)
        self.assertEqual(parts, ["The value is 1.5 kg.", "Done."])

    def test_caps_abbrev(self) -> None:
        text = "The U.S. Army arrived. It stayed."
        parts = split_english(text, min_len=1)
        self.assertEqual(parts, ["The U.S. Army arrived.", "It stayed."])

    def test_trailing_quote(self) -> None:
        text = 'Annina will pay him the silver in Kanesh."'
        parts = split_english(text, min_len=1)
        self.assertEqual(parts, ["Annina will pay him the silver in Kanesh."])

    def test_trailing_double_quote(self) -> None:
        text = 'Buy a .""'
        parts = split_english(text, min_len=1)
        self.assertEqual(parts, ["Buy a ."])

    def test_balanced_quote(self) -> None:
        text = 'He said: "Hello."'
        parts = split_english(text, min_len=1)
        self.assertEqual(parts, ['He said: "Hello."'])


if __name__ == "__main__":
    unittest.main()
