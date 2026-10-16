import hashlib
import json
from pathlib import Path


def read_rows(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = []
        for line in text.splitlines():
            if line.strip() != "":
                rows.append(json.loads(line))
        return rows
    return json.loads(text)


def digest(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def label(winner, position):
    if winner == "tie" or winner == "unknown":
        return winner
    if winner == position:
        return "win"
    if winner in ["A", "B"]:
        return "loss"
    raise ValueError("Invalid winner: " + str(winner))


def main():
    root = Path(__file__).resolve().parent
    results = {}
    for name in ["uf", "hh", "persona"]:
        pairs = read_rows(root / "outputs/generations/pairs" / (name + ".jsonl"))
        judgments = read_rows(root / "outputs/evaluation" / (name + "_judged.jsonl"))
        saved = read_rows(root / "outputs/evaluation" / (name + "_judged.summary.json"))
        metadata = read_rows(root / "outputs/evaluation" / (name + "_judged.meta.json"))
        if digest(pairs) != metadata["pairs_sha256"]:
            raise ValueError("Pair-file hash mismatch: " + name)

        by_id = {}
        for row in judgments:
            if row["id"] in by_id:
                raise ValueError("Duplicate judgment ID: " + row["id"])
            by_id[row["id"]] = row

        pair_ids = set()
        for pair in pairs:
            if pair["id"] in pair_ids:
                raise ValueError("Duplicate pair ID: " + pair["id"])
            pair_ids.add(pair["id"])
        if pair_ids != set(by_id):
            raise ValueError("Pair and judgment IDs differ: " + name)

        counts = {"win": 0, "loss": 0, "tie": 0, "unknown": 0, "failure": 0}
        axes = {}
        for axis in ["style_winner", "usefulness_winner", "safety_winner"]:
            axes[axis] = {"win": 0, "loss": 0, "tie": 0, "unknown": 0}
        identical = 0
        generations = {}
        for key in ["baseline", "candidate"]:
            model_name = pairs[0]["_metadata"][key]
            rows = read_rows(root / "outputs/generations" / (model_name + ".json"))
            generations[key] = {row["id"]: row for row in rows}

        for pair in pairs:
            row = by_id[pair["id"]]
            payload = dict(pair)
            recorded_hash = payload.pop("pair_sha256")
            if digest(payload) != recorded_hash:
                raise ValueError("Pair content changed: " + pair["id"])
            if row["pair_sha256"] != recorded_hash:
                raise ValueError("Judgment belongs to another pair: " + pair["id"])
            position = pair["_metadata"]["candidate_position"]
            baseline = generations["baseline"][pair["id"]]
            candidate = generations["candidate"][pair["id"]]
            for field in ["instruction", "generation_settings", "formatted_prompt", "chat_template_used"]:
                if baseline[field] != candidate[field]:
                    raise ValueError("Generation comparison mismatch: " + pair["id"])
            if pair["instruction"] != baseline["instruction"]:
                raise ValueError("Pair instruction differs from generation: " + pair["id"])
            candidate_key = "response_a" if position == "A" else "response_b"
            baseline_key = "response_b" if position == "A" else "response_a"
            if pair[candidate_key] != candidate["output"] or pair[baseline_key] != baseline["output"]:
                raise ValueError("Hidden mapping differs from generation: " + pair["id"])
            counts[label(row["winner"], position)] += 1
            for axis in axes:
                axes[axis][label(row.get(axis, "unknown"), position)] += 1
            if pair["response_a"] == pair["response_b"]:
                identical += 1

        if counts != saved["counts"] or axes != saved["separate_axes"]:
            raise ValueError("Saved summary disagrees with judgments: " + name)
        decisive = counts["win"] + counts["loss"]
        valid = decisive + counts["tie"]
        excluding = None
        half = None
        if decisive > 0:
            excluding = counts["win"] / decisive
        if valid > 0:
            half = (counts["win"] + 0.5 * counts["tie"]) / valid
        if excluding != saved["win_rate_excluding_ties"]:
            raise ValueError("Decisive win rate disagrees: " + name)
        if half != saved["win_rate_ties_half"]:
            raise ValueError("Half-tie win rate disagrees: " + name)

        requests = read_rows(root / "outputs/evaluation" / (name + "_judged.requests.jsonl"))
        failed_requests = 0
        for request in requests:
            if not request["success"]:
                failed_requests += 1
        results[name] = {
            "total_pairs": len(pairs),
            "counts": counts,
            "separate_axes": axes,
            "win_rate_excluding_ties": excluding,
            "win_rate_ties_half": half,
            "identical_response_pairs": identical,
            "requests_total": len(requests),
            "failed_request_attempts_total": failed_requests,
            "judge_model": metadata["judge_model"]
        }
        print(name + ": all judgment and summary checks passed")

    training = {}
    for name in ["sft", "dpo_hh", "dpo_uf", "persona"]:
        folder = root / "outputs" / name
        config = read_rows(folder / "config.json")
        complete = read_rows(folder / "COMPLETE.json")
        if complete.get("pilot"):
            raise ValueError("A pilot was supplied as the full run: " + name)
        metrics = read_rows(folder / "metrics.jsonl")
        if name == "persona":
            best_step = int(Path(complete["best_checkpoint"]).name.split("-")[-1])
            validation = []
            for row in metrics:
                if "eval_loss" in row and row["step"] == best_step:
                    validation.append(row)
            if len(validation) != 1:
                raise ValueError("Cannot locate persona best metrics")
            best = validation[0]
            if best["eval_rewards/accuracies"] != complete["best_metric"]:
                raise ValueError("Persona best metric mismatch")
        else:
            best = read_rows(folder / "best_metrics.json")
            validation = []
            train = []
            for row in metrics:
                if row.get("split") == "validation":
                    validation.append(row)
                if row.get("split") == "train":
                    train.append(row)
            if len(train) != complete["steps"]:
                raise ValueError("Training-step count mismatch: " + name)
            if name == "sft":
                selection = min(row["loss"] for row in validation if row["step"] > 0)
                if selection != best["loss"]:
                    raise ValueError("SFT checkpoint selection mismatch")
            else:
                selection = max(row["reward_accuracy"] for row in validation if row["step"] > 0)
                if selection != best["reward_accuracy"]:
                    raise ValueError("DPO checkpoint selection mismatch: " + name)
            matching = []
            for row in validation:
                if row["step"] == best["step"]:
                    matching.append(row)
            if len(matching) != 1:
                raise ValueError("Cannot locate selected validation metrics: " + name)
            for key in best:
                if matching[0][key] != best[key]:
                    raise ValueError("Best metrics differ from training log: " + name)
        training[name] = {
            "train_examples": config["train_examples"],
            "val_examples": config["val_examples"],
            "completion": complete,
            "selected_validation": best
        }
        print(name + ": training evidence checks passed")

    output = {"evaluation": results, "training": training}
    destination = root / "verified_results.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("Saved verified_results.json")


if __name__ == "__main__":
    main()
