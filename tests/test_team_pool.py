from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pokemon_battler.team_pool import ShuffledTeamPool, resolve_team_pool

TEAM_ONE = """Pikachu @ Light Ball
Ability: Static
Tera Type: Electric
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Thunderbolt
- Volt Switch
- Grass Knot
- Encore
"""

TEAM_TWO = """Raichu @ Heavy-Duty Boots
Ability: Lightning Rod
Tera Type: Water
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Thunderbolt
- Surf
- Nasty Plot
- Encore
"""

TEAM_THREE = """Zapdos @ Heavy-Duty Boots
Ability: Static
Tera Type: Steel
EVs: 248 HP / 244 Def / 16 Spe
Bold Nature
- Discharge
- Hurricane
- Roost
- Volt Switch
"""


class TeamPoolTests(unittest.TestCase):
    def _write(self, root: Path, name: str, team: str) -> Path:
        path = root / name
        path.write_text(team, encoding="utf-8")
        return path

    def test_pool_requires_two_distinct_team_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = self._write(root, "first.txt", TEAM_ONE)
            duplicate = self._write(root, "duplicate.txt", TEAM_ONE)
            with self.assertRaisesRegex(ValueError, "at least 2 distinct team compositions"):
                resolve_team_pool([first, duplicate])

    def test_reordering_one_composition_does_not_fake_team_diversity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = self._write(root, "first.txt", TEAM_ONE + "\n" + TEAM_TWO)
            reordered = self._write(root, "reordered.txt", TEAM_TWO + "\n" + TEAM_ONE)
            with self.assertRaisesRegex(
                ValueError, "at least 2 distinct team compositions"
            ):
                resolve_team_pool([first, reordered])

    def test_directory_pool_uses_every_team_before_repeating(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write(root, "one.txt", TEAM_ONE)
            self._write(root, "two.txt", TEAM_TWO)
            self._write(root, "three.txt", TEAM_THREE)
            paths = resolve_team_pool([], root)
            pool = ShuffledTeamPool(paths, seed=7)
            for _ in range(7):
                pool.yield_team()
            selections = [
                selection["team_index"] for selection in pool.report()["selections"]
            ]
            self.assertEqual(len(set(selections[:3])), 3)
            self.assertEqual(len(set(selections[3:6])), 3)
            self.assertTrue(
                all(left != right for left, right in zip(selections, selections[1:]))
            )

    def test_pool_selection_is_reproducible_for_a_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = [
                self._write(root, "one.txt", TEAM_ONE),
                self._write(root, "two.txt", TEAM_TWO),
                self._write(root, "three.txt", TEAM_THREE),
            ]
            first = ShuffledTeamPool(paths, seed=19)
            second = ShuffledTeamPool(paths, seed=19)
            for _ in range(9):
                first.yield_team()
                second.yield_team()
            self.assertEqual(first.report()["selections"], second.report()["selections"])


if __name__ == "__main__":
    unittest.main()
