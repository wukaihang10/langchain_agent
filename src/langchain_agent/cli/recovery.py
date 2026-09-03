from collections.abc import Sequence

from langchain_agent.app.session_continuation import (
    ToolCallResolution,
    ToolCallResolutionKind,
    UnresolvedToolCall,
)


def collect_tool_call_resolutions(
    calls: Sequence[UnresolvedToolCall],
) -> list[ToolCallResolution]:
    resolutions: list[ToolCallResolution] = []

    for call in calls:
        if not call.resolution_required:
            continue

        print("\n--- Uncertain tool outcome ---")
        print(f"Tool: {call.name}")
        print(f"Tool call ID: {call.id}")
        print(f"Arguments: {call.args}")
        print("1. Confirm succeeded")
        print("2. Confirm not applied")
        print("3. Retry despite risk")
        print("4. Record outcome unknown")

        while True:
            choice = input("Resolution [1-4]: ").strip()
            if choice == "1":
                summary = input("Optional verified result summary: ").strip()
                resolutions.append(
                    ToolCallResolution(
                        tool_call_id=call.id,
                        kind=ToolCallResolutionKind.CONFIRM_SUCCEEDED,
                        result_summary=summary or None,
                    )
                )
                break
            if choice == "2":
                note = input("Optional verification note: ").strip()
                resolutions.append(
                    ToolCallResolution(
                        tool_call_id=call.id,
                        kind=ToolCallResolutionKind.CONFIRM_NOT_APPLIED,
                        note=note or None,
                    )
                )
                break
            if choice == "3":
                print(
                    "Warning: retrying may cause a duplicate side effect if the "
                    "original external operation actually succeeded."
                )
                confirmation = input(
                    "Retry this tool despite the risk? [y/n]: "
                ).strip().lower()
                if confirmation not in {"y", "yes"}:
                    print(
                        "Risk-bearing retry not confirmed; choose another resolution."
                    )
                    continue
                resolutions.append(
                    ToolCallResolution(
                        tool_call_id=call.id,
                        kind=ToolCallResolutionKind.RETRY_DESPITE_RISK,
                    )
                )
                break
            if choice == "4":
                note = input("Optional note: ").strip()
                resolutions.append(
                    ToolCallResolution(
                        tool_call_id=call.id,
                        kind=ToolCallResolutionKind.RECORD_OUTCOME_UNKNOWN,
                        note=note or None,
                    )
                )
                break

            print("Invalid resolution. Choose 1, 2, 3, or 4.")

    return resolutions
