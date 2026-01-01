import sys
import unittest
from pathlib import Path

# テスト実行時に src を import できるようにパスを追加する
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dp.align_train import normalize_transliteration, normalize_translation


class TestPreprocessInvariants(unittest.TestCase):
    def test_gap_tokens_survive(self) -> None:
        text = "a <gap> b <big_gap> c"
        for variant in ("A", "B", "C"):
            out = normalize_transliteration(text, variant)
            self.assertIn("<gap>", out)
            self.assertIn("<big_gap>", out)

    def test_gap_normalization_from_brackets(self) -> None:
        text = "a [x] b [X] c [x?] d [...] e … f xxx g xxxx h"
        for variant in ("B", "C"):
            out = normalize_transliteration(text, variant)
            self.assertIn("<gap>", out)
            self.assertIn("<big_gap>", out)

    def test_spaced_x_tokens_to_gap(self) -> None:
        text = "a x x x b x x x x c"
        for variant in ("B", "C"):
            out = normalize_transliteration(text, variant)
            self.assertIn("<big_gap>", out)
            self.assertNotIn("x x x", out)

    def test_dots_to_big_gap(self) -> None:
        text = "a ... b .... c"
        for variant in ("B", "C"):
            out = normalize_transliteration(text, variant)
            self.assertIn("<big_gap>", out)
            self.assertNotIn("...", out)

    def test_hyphenated_x_tokens_to_gap(self) -> None:
        text = "a x-x-x b x- x -x c x-x-x-x d"
        for variant in ("B", "C"):
            out = normalize_transliteration(text, variant)
            self.assertIn("<big_gap>", out)
            self.assertNotIn("x-x-x", out)

    def test_cdli_oracc_normalization(self) -> None:
        text = "a2 a3 a₂ a₃ e2 e3 e₂ e₃ i2 i3 i₂ i₃ u2 u3 u₂ u₃ sz SZ s, S, s' S' t, T, Xx ʾ ʼ h H"
        for variant in ("A", "B", "C"):
            out = normalize_transliteration(text, variant)
            self.assertIn("á", out)
            self.assertIn("à", out)
            self.assertIn("é", out)
            self.assertIn("è", out)
            self.assertIn("í", out)
            self.assertIn("ì", out)
            self.assertIn("ú", out)
            self.assertIn("ù", out)
            self.assertIn("š", out)
            self.assertIn("Š", out)
            self.assertIn("ṣ", out)
            self.assertIn("Ṣ", out)
            self.assertIn("ś", out)
            self.assertIn("Ś", out)
            self.assertIn("ṭ", out)
            self.assertIn("Ṭ", out)
            self.assertIn("ₓ", out)
            self.assertIn("ḫ", out)
            self.assertIn("Ḫ", out)
            self.assertIn("'", out)
            self.assertNotIn("sz", out.lower())
            self.assertNotIn("s,", out)
            self.assertNotIn("s'", out)
            self.assertNotIn("t,", out)
            self.assertNotIn("Xx", out)

    def test_brace_big_gap(self) -> None:
        text = "a {large break} b {broken area} c"
        for variant in ("B", "C"):
            out = normalize_transliteration(text, variant)
            self.assertGreaterEqual(out.count("<big_gap>"), 2)

    def test_parenthetical_big_gap(self) -> None:
        text = "a (broken lines) b (illegible) c"
        for variant in ("B", "C"):
            out = normalize_transliteration(text, variant)
            self.assertGreaterEqual(out.count("<big_gap>"), 2)

    def test_parenthetical_text_removed(self) -> None:
        text = "a (foo) b"
        for variant in ("A", "B", "C"):
            out = normalize_transliteration(text, variant)
            self.assertIn("a", out)
            self.assertIn("b", out)
            self.assertNotIn("foo", out)

    def test_logogram_dot_preserved(self) -> None:
        text = "KÙ.BABBAR"
        for variant in ("A", "B", "C"):
            out = normalize_transliteration(text, variant)
            self.assertIn("KÙ.BABBAR", out)

    def test_standalone_dot_removed(self) -> None:
        text = "a . b"
        for variant in ("A", "B", "C"):
            out = normalize_transliteration(text, variant)
            self.assertNotIn(".", out)

    def test_all_caps_preserved(self) -> None:
        text = "URUK LUGAL"
        for variant in ("A", "B", "C"):
            out = normalize_transliteration(text, variant)
            self.assertIn("URUK", out)
            self.assertIn("LUGAL", out)

    def test_title_case_preserved(self) -> None:
        text = "Aššur aššur URUK"
        for variant in ("A", "B", "C"):
            out = normalize_transliteration(text, variant)
            self.assertIn("Aššur", out)
            self.assertIn("aššur", out)
            self.assertIn("URUK", out)

    def test_title_case_hyphen_preserved_in_c(self) -> None:
        text = "A-šur a-šur"
        out = normalize_transliteration(text, "C")
        self.assertIn("A-šur", out)
        self.assertNotIn("A šur", out)
        self.assertNotIn("a-šur", out)

    def test_determinatives_to_curly(self) -> None:
        text = "(d)EN"
        out_a = normalize_transliteration(text, "A")
        out_b = normalize_transliteration(text, "B")
        out_c = normalize_transliteration(text, "C")
        self.assertIn("(d)EN", out_a)
        self.assertIn("{d}EN", out_b)
        self.assertIn("{d}EN", out_c)

    def test_determinative_tug_to_curly(self) -> None:
        text = "(TÚG)ku-ta-nu"
        out_a = normalize_transliteration(text, "A")
        out_b = normalize_transliteration(text, "B")
        out_c = normalize_transliteration(text, "C")
        self.assertIn("(TÚG)ku-ta-nu", out_a)
        self.assertIn("{TÚG}ku-ta-nu", out_b)
        self.assertIn("{TÚG}ku ta nu", out_c)

    def test_html_tags_are_removed(self) -> None:
        text = "a <sup>1</sup> b"
        for variant in ("B", "C"):
            out = normalize_transliteration(text, variant)
            self.assertIn("1", out)
            self.assertNotIn("sup", out.lower())

    def test_html_sup_determinative_to_brace(self) -> None:
        text = "a <sup>d</sup>UTU b"
        out = normalize_transliteration(text, "B")
        self.assertIn("{d}UTU", out)

    def test_double_angle_brackets_removed(self) -> None:
        text = "a <<x>> b"
        out = normalize_transliteration(text, "B")
        self.assertIn("a", out)
        self.assertIn("b", out)
        self.assertNotIn("x", out)
        self.assertNotIn("<", out)
        self.assertNotIn(">", out)

    def test_oracc_excision_hyphen_collapse(self) -> None:
        text = "mu-un-<<an>>-pa3-da"
        out = normalize_transliteration(text, "B")
        self.assertIn("mu-un-pà-da", out)
        self.assertNotIn("- -", out)

    def test_oracc_slash_alternative_split(self) -> None:
        text = "KI/DI-bi"
        out = normalize_transliteration(text, "B")
        self.assertIn("KI DI-bi", out)
        self.assertNotIn("KIDI-bi", out)

    def test_oracc_gloss_shift_removed(self) -> None:
        text = "AN{+e} du₃-am₃{{mu-un-<du₃>}} {(1(u))} %akk a"
        out = normalize_transliteration(text, "B")
        self.assertIn("AN{e}", out)
        self.assertIn("dù-am3", out)
        self.assertIn("a", out)
        self.assertNotIn("{{", out)
        self.assertNotIn("{(", out)
        self.assertNotIn("%akk", out)

    def test_oracc_doc_gloss_nested_removed(self) -> None:
        text = "a {(1(u))} b"
        out = normalize_transliteration(text, "B")
        self.assertIn("a", out)
        self.assertIn("b", out)
        self.assertNotIn("1(u)", out)
        self.assertNotIn("{", out)
        self.assertNotIn("}", out)

    def test_oracc_punct_flags_removed(self) -> None:
        text = "a *(KUR) ba# /(P2) c |GA₂×(ME.EN)| : :' :: :."
        out = normalize_transliteration(text, "B")
        self.assertIn("a", out)
        self.assertIn("ba", out)
        self.assertIn("c", out)
        self.assertIn("GÁ×ME.EN", out)
        self.assertNotIn("KUR", out)
        self.assertNotIn("P2", out)
        self.assertNotIn("|", out)
        self.assertNotIn("#", out)
        self.assertNotIn(":'", out)

    def test_oracc_markers_removed(self) -> None:
        text = "a$1 $AN EN~a N07~a@h (#note#) a; b // c"
        out = normalize_transliteration(text, "B")
        self.assertIn("a", out)
        self.assertIn("AN", out)
        self.assertIn("EN", out)
        self.assertIn("N07", out)
        self.assertIn("b", out)
        self.assertIn("c", out)
        self.assertNotIn("$", out)
        self.assertNotIn("~", out)
        self.assertNotIn("@", out)
        self.assertNotIn("#note#", out)
        self.assertNotIn(";", out)
        self.assertNotIn("//", out)

    def test_line_number_removed(self) -> None:
        text = "1' a 2'' b"
        out = normalize_transliteration(text, "B")
        self.assertNotIn("1'", out)
        self.assertNotIn("2''", out)

    def test_translation_brackets_removed(self) -> None:
        text = 'note <add> [missing] text'
        out = normalize_translation(text)
        self.assertIn("add", out)
        self.assertIn("missing", out)
        self.assertNotIn("<", out)
        self.assertNotIn(">", out)
        self.assertNotIn("[", out)
        self.assertNotIn("]", out)

    def test_translation_slash_tokens(self) -> None:
        text = "a / b 1 / 4 3/5"
        out = normalize_translation(text)
        tokens = out.split()
        self.assertEqual(tokens[:2], ["a", "b"])
        self.assertIn("1 / 4", out)
        self.assertIn("3/5", out)


if __name__ == "__main__":
    unittest.main()
