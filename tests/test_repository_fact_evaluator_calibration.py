import unittest

from evals.calibration.repository_fact import (
    load_repository_fact_calibration_cases,
)


def load_calibration_cases() -> list[dict]:
    return load_repository_fact_calibration_cases()


class RepositoryFactEvaluatorCalibrationTests(unittest.TestCase):
    def test_calibration_cases_cover_the_agreed_boundaries(self):
        cases = load_calibration_cases()

        self.assertEqual(len(cases), 11)
        self.assertEqual(
            [case["metadata"]["case_id"] for case in cases],
            [f"repository_fact_evaluator_{number:03d}" for number in range(1, 12)],
        )
        self.assertEqual(
            [case["metadata"]["case_type"] for case in cases],
            [
                "missing_required_fact",
                "forbidden_claim",
                "missing_evidence",
                "fabricated_evidence",
                "trace_unavailable",
                "non_material_imprecision",
                "approximate_location",
                "evidence_conflict",
                "partial_sufficient_evidence",
                "partial_insufficient_evidence",
                "partial_evidence_conflict",
            ],
        )
        self.assertEqual(
            [case["expected_feedback"] for case in cases],
            [
                {
                    "answer_correctness": "fail",
                    "evidence_groundedness": "pass",
                },
                {
                    "answer_correctness": "fail",
                    "evidence_groundedness": "fail",
                },
                {
                    "answer_correctness": "pass",
                    "evidence_groundedness": "fail",
                },
                {
                    "answer_correctness": "pass",
                    "evidence_groundedness": "fail",
                },
                {
                    "answer_correctness": "pass",
                    "evidence_groundedness": "unknown",
                },
                {
                    "answer_correctness": "pass",
                    "evidence_groundedness": "pass",
                },
                {
                    "answer_correctness": "pass",
                    "evidence_groundedness": "pass",
                },
                {
                    "answer_correctness": "pass",
                    "evidence_groundedness": "fail",
                },
                {
                    "answer_correctness": "pass",
                    "evidence_groundedness": "pass",
                },
                {
                    "answer_correctness": "pass",
                    "evidence_groundedness": "unknown",
                },
                {
                    "answer_correctness": "pass",
                    "evidence_groundedness": "fail",
                },
            ],
        )

    def test_calibration_cases_match_the_fixture_contract(self):
        cases = load_calibration_cases()

        for case in cases:
            with self.subTest(case=case["metadata"]["case_id"]):
                self.assertEqual(
                    set(case),
                    {
                        "inputs",
                        "outputs",
                        "reference_outputs",
                        "trace",
                        "expected_feedback",
                        "metadata",
                        "rationale",
                    },
                )
                self.assertEqual(set(case["inputs"]), {"question"})
                self.assertEqual(set(case["outputs"]), {"answer"})
                self.assertEqual(
                    set(case["reference_outputs"]),
                    {
                        "required_facts",
                        "forbidden_claims",
                        "acceptable_evidence",
                    },
                )
                self.assertEqual(
                    set(case["trace"]),
                    {"status", "evidence"},
                )
                self.assertIn(
                    case["trace"]["status"],
                    {"available", "partial", "unavailable"},
                )
                self.assertIsInstance(case["trace"]["evidence"], list)
                self.assertEqual(
                    set(case["expected_feedback"]),
                    {"answer_correctness", "evidence_groundedness"},
                )
                self.assertEqual(
                    set(case["metadata"]),
                    {"slice", "case_type", "case_id"},
                )
                self.assertEqual(
                    case["metadata"]["slice"],
                    "repository_fact_evaluator",
                )
                self.assertTrue(case["rationale"].strip())

                for verdict in case["expected_feedback"].values():
                    self.assertIn(verdict, {"pass", "fail", "unknown"})
                for evidence in case["trace"]["evidence"]:
                    self.assertEqual(
                        set(evidence),
                        {"tool_name", "inputs", "output", "error"},
                    )

    def test_unavailable_trace_is_distinct_from_available_empty_evidence(self):
        cases = load_calibration_cases()
        by_type = {case["metadata"]["case_type"]: case for case in cases}

        missing_evidence = by_type["missing_evidence"]
        unavailable_trace = by_type["trace_unavailable"]

        self.assertEqual(missing_evidence["trace"]["status"], "available")
        self.assertEqual(missing_evidence["trace"]["evidence"], [])
        self.assertEqual(
            missing_evidence["expected_feedback"]["evidence_groundedness"],
            "fail",
        )
        self.assertEqual(unavailable_trace["trace"]["status"], "unavailable")
        self.assertEqual(unavailable_trace["trace"]["evidence"], [])
        self.assertEqual(
            unavailable_trace["expected_feedback"]["evidence_groundedness"],
            "unknown",
        )

    def test_partial_trace_covers_all_three_groundedness_verdicts(self):
        cases = load_calibration_cases()
        partial_cases = [case for case in cases if case["trace"]["status"] == "partial"]

        self.assertEqual(
            [
                case["expected_feedback"]["evidence_groundedness"]
                for case in partial_cases
            ],
            ["pass", "unknown", "fail"],
        )


if __name__ == "__main__":
    unittest.main()
