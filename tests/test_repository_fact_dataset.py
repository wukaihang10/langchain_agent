import copy
import unittest

from evals.datasets.repository_fact import (
    load_repository_fact_dataset,
    validate_repository_fact_examples,
)


class RepositoryFactDatasetTests(unittest.TestCase):
    def test_source_matches_the_accepted_contract(self):
        examples = load_repository_fact_dataset()

        self.assertEqual(len(examples), 8)
        self.assertEqual(
            [example["metadata"]["case_id"] for example in examples],
            [f"repository_fact_{number:03d}" for number in range(1, 9)],
        )
        self.assertTrue(
            all(set(example["inputs"]) == {"question"} for example in examples)
        )
        self.assertTrue(
            all(
                set(example["metadata"]) == {"slice", "case_type", "case_id"}
                for example in examples
            )
        )

    def test_rejects_duplicate_case_ids(self):
        examples = load_repository_fact_dataset()
        duplicate = copy.deepcopy(examples[1])
        duplicate["metadata"]["case_id"] = examples[0]["metadata"]["case_id"]

        with self.assertRaisesRegex(ValueError, "duplicate repository-fact case_id"):
            validate_repository_fact_examples([examples[0], duplicate])

    def test_rejects_case_ids_outside_the_slice_number_format(self):
        example = copy.deepcopy(load_repository_fact_dataset()[0])
        example["metadata"]["case_id"] = "unknown-policy"

        with self.assertRaisesRegex(
            ValueError,
            r"must match repository_fact_\[0-9\]\{3\}",
        ):
            validate_repository_fact_examples([example])


if __name__ == "__main__":
    unittest.main()
