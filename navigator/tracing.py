"""Optional fail-open JSONL sink. Persist a structural allowlist, never raw text."""
import json
import math
import re
from pathlib import Path

REDACTED = "[REDACTED]"
OPERATIONS = set("CLICK TYPE_TEXT SELECT SCROLL_DOWN SCROLL_UP WAIT DONE BLOCKED".split())
FLAGS = set("step_completed expected_state_reached unexpected_change human_review_required".split())
EVENTS = set("run_started run_finished node_started node_finished step_started step_finished step_context observation jev_request decision policy action_selected action_attempt_started action_attempt_error action_attempt_finished action acceptance outputs escalation error native_dialog trace_error".split())
TEXT = set("goal effective_goal subgoal reason error message title modal fingerprint input_name input_text readback detail model describe diagnosis".split())
NUMBERS = set("schema_version step sequence t element_count latency_ms input_tokens attempt retry_count ms total_step_ms".split())
BOOLS = set("ok auto retry_safe page_changed tokens_measured".split())
IDS = set("run_id procedure_id version node node_execution_id step_id verifies_step_id".split())
METRICS = set("elapsed_s jev_requests median_latency_ms input_tokens jev_cost_usd escalations escalation_tokens steps verified".split())
ENUMS = {
    "type": EVENTS,
    "status": set("success error interrupted handoff blocked DONE BLOCKED HANDOFF continue".split()),
    "kind": set("ui_task data_task decision end".split()),
    "phase": {"initial", "after_action"},
    "stage": set("decide policy guards act readback acceptance capture escalate".split()),
    "execution_status": {"stale", "error", "returned"},
    "dialog_type": {"alert", "confirm", "prompt", "beforeunload"},
    "error_type": set("JevError StaleTarget Error TimeoutError RuntimeError ValueError TypeError KeyError AttributeError OSError PermissionError FileNotFoundError BrokenPipeError CancelledError KeyboardInterrupt SystemExit".split()),
}
CHECKS = {"url_matches", "text_visible", "dialog_visible", "field_equals"}
CHECKS |= {"guard " + name for name in CHECKS}
SENSITIVE = re.compile(r"authorization|cookie|password|passwd|secret|token|api.?key|credential|bearer|sk-", re.I)


def observation_summary(obs):
    return {"url": obs.get("url"), "title": obs.get("title"), "modal": obs.get("modal"),
            "element_count": len(obs.get("elements", [])), "fingerprint": obs.get("fingerprint")}


def number(value, probability=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (not probability or 0 <= value <= 1))


def masked(value):
    return None if value is None else REDACTED


def url_scheme(value):
    # Even a hostname can contain customer IDs or credentials.
    if isinstance(value, str):
        scheme = value.split(":", 1)[0].lower()
        if scheme in {"http", "https", "file", "about", "data", "blob"}:
            return scheme + ":" + REDACTED
    return masked(value)


class TraceWriter:
    def __init__(self, directory, run_id, secrets=()):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
            raise ValueError("invalid trace run ID")
        self.path = Path(directory) / run_id / "execution_trace.jsonl"
        self.secrets = tuple(str(s) for s in secrets if s)
        self.failed = False
        self.offered_step = None
        self.operations, self.targets = set(), {}

    def _identifier(self, value):
        if value is None:
            return None
        if (type(value) is not str or not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,160}", value)
                or SENSITIVE.search(value) or any(s in value for s in self.secrets)):
            return REDACTED
        return value

    def _remember_offers(self, event):
        if event.get("type") in {"node_started", "step_started", "jev_request"}:
            self.operations, self.targets = set(), {}
            self.offered_step = event.get("step_id")
        if event.get("type") != "jev_request":
            return
        request = event.get("request")
        questions = request.get("questions", {}) if isinstance(request, dict) else {}
        if not isinstance(questions, dict):
            return
        def criteria(name):
            question = questions.get(name)
            values = question.get("criteria") if isinstance(question, dict) else None
            return values if isinstance(values, dict) else {}
        self.operations = set(criteria("operation")) & OPERATIONS
        for op in self.operations:
            self.targets[op] = {key for key in criteria(op.lower() + "_target")
                                if type(key) is str and re.fullmatch(r"[1-9][0-9]*(?::[1-9][0-9]*)?", key)}

    @staticmethod
    def _choice(choice, allowed):
        if not isinstance(choice, dict):
            return None
        result = {}
        value = choice.get("value")
        if type(value) is str and value in allowed:
            result["value"] = value
        if number(choice.get("p"), probability=True):
            result["p"] = choice["p"]
        dist = choice.get("dist")
        if isinstance(dist, dict):
            result["dist"] = {k: v for k, v in dist.items()
                              if k in allowed and number(v, probability=True)}
        return result

    def project(self, event):
        self._remember_offers(event)
        data = {}
        for key, value in event.items():
            if key in TEXT:
                data[key] = masked(value)
            elif key in IDS:
                data[key] = self._identifier(value)
            elif key in NUMBERS and number(value):
                data[key] = value
            elif key in BOOLS and type(value) is bool:
                data[key] = value
            elif key in ENUMS:
                data[key] = value if type(value) is str and value in ENUMS[key] else "unknown"
            elif key in {"url", "url_before", "url_after"}:
                data[key] = url_scheme(value)
        summary = event.get("observation_summary")
        if isinstance(event.get("obs"), dict):
            summary = observation_summary(event["obs"])
        if isinstance(summary, dict):
            data["observation_summary"] = {k: masked(summary[k]) for k in ("title", "modal", "fingerprint") if k in summary}
            if "url" in summary:
                data["observation_summary"]["url"] = url_scheme(summary["url"])
            if number(summary.get("element_count")):
                data["observation_summary"]["element_count"] = summary["element_count"]
        offered = self.operations if event.get("step_id") == self.offered_step else set()
        operation = event.get("operation")
        op = operation.get("value") if isinstance(operation, dict) else None
        targets = self.targets.get(op, set()) if type(op) is str and op in offered else set()
        for key, allowed in (("operation", offered), ("target", targets)):
            if key in event:
                data[key] = self._choice(event[key], allowed)
        # Input-variable names and their distributions are free text too.
        if "value" in event:
            data["value"] = masked(event["value"])
        flags = event.get("flags")
        if isinstance(flags, dict):
            data["flags"] = {k: v for k, v in flags.items() if k in FLAGS and (v is None or number(v, True))}
        verdict = event.get("verdict")
        if isinstance(verdict, dict):
            code = verdict.get("verdict")
            data["verdict"] = {"verdict": code if type(code) is str and code in {"ACT", "ESCALATE", "HANDOFF", "BLOCKED", "DONE_CHECK"} else "unknown",
                               "reason": masked(verdict.get("reason"))}
        row = event.get("selected_target")
        if isinstance(row, dict):
            data["selected_target"] = {"label": masked(row.get("label")), "section": masked(row.get("section"))}
            if type(row.get("index")) is str and row["index"] in {t.split(":")[0] for t in targets}:
                data["selected_target"]["index"] = row["index"]
            role = row.get("role")
            data["selected_target"]["role"] = role if type(role) is str and role in {"button", "link", "textbox", "combobox", "checkbox", "radio", "option", "menuitem", "spinbutton", "searchbox"} else "unknown"
            option = row.get("selected_option")
            if isinstance(option, dict) and type(option.get("index")) is str and option["index"] in targets:
                data["selected_target"]["selected_option"] = {"index": option["index"], "label": masked(option.get("label"))}
        metrics = event.get("metrics")
        if isinstance(metrics, dict):
            data["metrics"] = {k: v for k, v in metrics.items() if k in METRICS and number(v)}
            if type(metrics.get("tokens_measured")) is bool:
                data["metrics"]["tokens_measured"] = metrics["tokens_measured"]
            buckets = metrics.get("time_buckets_s")
            if isinstance(buckets, dict):
                data["metrics"]["time_buckets_s"] = {k: v for k, v in buckets.items() if k in {"decide", "act", "wait", "escalate"} and number(v)}
        results = event.get("results")
        if isinstance(results, list):
            data["results"] = [{"check": r["check"], "ok": r["ok"], "detail": masked(r.get("detail"))}
                               for r in results if isinstance(r, dict) and type(r.get("check")) is str
                               and r["check"] in CHECKS and type(r.get("ok")) is bool]
        return data

    def write(self, event):
        """Append and flush each event. Failure must never affect automation."""
        if self.failed:
            return False
        try:
            line = json.dumps(self.project(event), ensure_ascii=True, allow_nan=False) + "\n"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()
            return True
        except Exception:
            self.failed = True
            return False
