def collect_hitl_decisions(interrupts) -> list[dict]:
    decisions: list[dict] = []

    for interrupt_item in interrupts:
        value = interrupt_item.value
        action_requests = value.get("action_requests", [])
        review_configs = value.get("review_configs", [])

        for action, review_config in zip(
            action_requests,
            review_configs,
            strict=True,
        ):
            print("\n--- Tool approval required ---")
            # print("Tool:", action["name"])
            # print("Arguments:", action["args"])

            description = action.get("description")

            if description:
                print("Reason:", description)

            allowed = review_config["allowed_decisions"]
            print("Allowed:", ", ".join(allowed))

            while True:
                answer = input("Approve? [y/n]: ").strip().lower()

                if answer in {"y", "yes"} and "approve" in allowed:
                    decisions.append({"type": "approve"})
                    break

                if answer in {"n", "no"} and "reject" in allowed:
                    message = input("Optional rejection reason: ").strip()
                    decision = {"type": "reject"}

                    if message:
                        decision["message"] = message

                    decisions.append(decision)
                    break

                print("Invalid decision. " f"Allowed: {allowed}")

    return decisions
