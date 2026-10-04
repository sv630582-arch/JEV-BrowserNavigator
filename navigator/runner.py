"""Runs an AOP workflow_graph. Node kinds: ui_task, data_task, decision, end.
The demo runner stands in for LangGraph; run_ui_task() is the node function LangGraph will wrap."""
import asyncio
import json
import queue
import re
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .agent import Metrics, run_ui_task
from .browser import Browser
from .tracing import TraceWriter

VAR = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")


def render(obj, variables: dict):
    if isinstance(obj, str):
        whole = VAR.fullmatch(obj.strip())
        if whole:  # a lone {{var}} keeps its type (lists, dicts)
            return _lookup(variables, whole.group(1), obj)
        return VAR.sub(lambda m: str(_lookup(variables, m.group(1), m.group(0))), obj)
    if isinstance(obj, list):
        return [render(x, variables) for x in obj]
    if isinstance(obj, dict):
        return {k: render(v, variables) for k, v in obj.items()}
    return obj


def _lookup(variables, path, default):
    cur = variables
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


class EventBus:
    """Fan-out of run events to any number of SSE listeners (thread-safe)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.listeners: list[queue.Queue] = []
        self.history: list[dict] = []
        self.trace_writer = None
        self.sequence = 0
        self._trace_failed = False

    def configure_trace(self, cfg, run_id, secrets=()):
        self.trace_writer = None
        self._trace_failed = False
        try:
            if cfg.get("enabled", False):
                self.trace_writer = TraceWriter(cfg.get("directory", "out/traces"), run_id, secrets)
        except Exception:
            self.emit("trace_error", {"run_id": run_id, "message": "Invalid trace configuration; automation continues."})

    @staticmethod
    def _print(type_, data):
        try:
            brief = {k: v for k, v in data.items() if k not in ("obs", "request", "raw_answers", "metrics")}
            print(f"[{type_}] {json.dumps(brief, default=str)[:300]}", flush=True)
        except Exception:
            pass  # A closed console is not an automation failure.

    def _publish(self, event):
        self.history.append(event)  # Delivery means accepted into the replayable bus history.
        for listener in self.listeners:
            try:
                listener.put_nowait(event)
            except Exception:
                pass

    def terminal_delivered(self, run_id):
        with self.lock:
            return any(e["type"] == "run_finished" and e.get("run_id") == run_id for e in self.history)

    def emit(self, type_: str, data: dict):
        try:
            ev = {**data, "type": type_, "t": time.time()}
            self._print(type_, data)
            with self.lock:
                if type_ == "run_finished" and any(e["type"] == type_ and e.get("run_id") == ev.get("run_id") for e in self.history):
                    return True
                if self.trace_writer is not None:
                    self.sequence += 1
                    ev["sequence"] = self.sequence
                self._publish(ev)
                if self.trace_writer is not None and not self._trace_failed:
                    try:
                        written = self.trace_writer.write(ev)
                    except Exception:
                        written = False
                    if not written:
                        self._trace_failed = True
                        self.sequence += 1
                        warning = {k: ev[k] for k in ("schema_version", "run_id", "node_execution_id", "step_id", "step") if k in ev}
                        warning.update(type="trace_error", t=time.time(), sequence=self.sequence,
                                       message="Trace persistence failed; automation continues.")
                        self._print("trace_error", {"message": warning["message"]})
                        self._publish(warning)
            return True
        except Exception:
            return False  # Telemetry only; automation exceptions never enter this method.

    def subscribe(self) -> queue.Queue:
        q = queue.Queue()
        with self.lock:
            for ev in self.history:
                q.put(ev)
            self.listeners.append(q)
        return q

    def reset(self):
        with self.lock:
            self.history.clear()
            self.sequence = 0
            self.trace_writer = None


class Runtime:
    def __init__(self, cfg, keys, bus: EventBus, frames=None, run_id=None):
        self.cfg, self.keys, self.bus = cfg, keys, bus
        self.run_id = run_id or uuid4().hex
        self.tracing = bool(cfg.get("tracing", {}).get("enabled", False))
        self.bus.configure_trace(cfg.get("tracing", {}), self.run_id, keys.values())
        self.context = {"schema_version": 1, "run_id": self.run_id}
        self.node_execution = 0
        self.step_id = self.previous_action_step_id = None
        self.step_outcome = None
        self.finished = False
        self.terminal_attempted = False
        self._terminal_payload = None
        self._terminal_lock = threading.Lock()
        self._task_context_claimed = False
        self.metrics = Metrics()
        self.browser = Browser(cfg.get("browser", {}), frames, on_event=self.emit)

    def emit(self, type_, data):
        if type_ == "run_finished":
            with self._terminal_lock:
                if self.finished:
                    return True
                self.terminal_attempted = True
                if self._terminal_payload is None:
                    self._terminal_payload = dict(data)
                delivered = self._emit(type_, {**self._terminal_payload, "run_id": self.run_id})
                # A bus may accept the event before its acknowledgement fails.
                try:
                    self.finished = delivered or self.bus.terminal_delivered(self.run_id)
                except Exception:
                    self.finished = delivered
                return self.finished
        return self._emit(type_, data)

    def _emit(self, type_, data):
        try:
            if self.tracing:
                if type_ == "run_started":
                    self.context.update(procedure_id=data.get("procedure_id"), version=data.get("version"))
                if type_ == "node_started":
                    self.node_execution += 1
                    self.context.update(node=data["node"], node_execution_id=f"{self.run_id}:n{self.node_execution}", goal=data.get("goal"))
                    self.previous_action_step_id = None
                    self._task_context_claimed = False
                data = {**self.context, **data}
                if self.step_id:
                    data.update(step_id=self.step_id, step=self.step_number)
                    stages = {"step_context": "decide", "decision": "policy", "action_selected": "guards",
                              "action_attempt_started": "act", "action": "readback", "acceptance": "acceptance",
                              "outputs": "capture", "escalation": "escalate"}
                    if type_ in stages:
                        self.step_outcome["stage"] = stages[type_]
                if type_ == "action":
                    self.previous_action_step_id = self.step_id
            return self.bus.emit(type_, data) is True
        except Exception:
            return False

    def finish_run(self, data):
        # Bounded event-only fallback. Never retry the workflow or its actions.
        for _ in range(2):
            try:
                if self.emit("run_finished", data):
                    return True
            except Exception:
                pass
        return False

    def ensure_task_context(self, node):
        if not self.tracing:
            return
        try:
            if "node_execution_id" not in self.context or self._task_context_claimed:
                self.emit("node_started", {"node": "standalone_ui_task", "kind": "ui_task", "goal": node.get("goal")})
            self._task_context_claimed = True
        except Exception:
            pass

    @contextmanager
    def step_scope(self, step):
        if not self.tracing:
            yield
            return
        start = None
        try:
            if "node_execution_id" not in self.context:
                self.ensure_task_context({})
            self.step_number = step
            self.step_id = f"{self.context['node_execution_id']}:s{step}"
            self.step_outcome = {"status": "continue"}
            start = time.perf_counter()
            self.emit("step_started", {})
        except Exception:
            pass
        try:
            yield  # Only exceptions from the automation body propagate.
        except BaseException as e:
            try:
                if self.step_outcome is not None:
                    self.step_outcome.update(status=exception_status(e), error_type=type(e).__name__)
            except Exception:
                pass
            raise
        finally:
            try:
                if start is not None and self.step_outcome is not None:
                    self.emit("step_finished", {**self.step_outcome, "total_step_ms": round((time.perf_counter() - start) * 1000)})
            except Exception:
                pass
            finally:
                self.step_id = None
                self.step_outcome = None

    def task_result(self, result):
        if self.step_outcome is not None:
            self.step_outcome.update(status=result["status"], reason=result["reason"])
        return result

    def trace_event(self, type_, data):
        if self.tracing:
            self.emit(type_, data)


# ---------------- data tasks (deterministic tools) ----------------
def write_excel(data, path, meta: dict) -> str:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "Captured fields"
    head_font, head_fill = Font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor="16181A")
    ws.append(["Field", "Value (as displayed)", "Note"])
    for c in ws[1]:
        c.font, c.fill = head_font, head_fill
    for r in data or []:
        ws.append([r.get("field"), r.get("value"), r.get("note")])
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width, ws.column_dimensions["C"].width = 24, 24, 70
    m = wb.create_sheet("Run metadata")
    m.append(["Key", "Value"])
    for c in m[1]:
        c.font, c.fill = head_font, head_fill
    for k, v in meta.items():
        m.append([k, json.dumps(v) if isinstance(v, (dict, list)) else v])
    m.column_dimensions["A"].width, m.column_dimensions["B"].width = 28, 90
    wb.save(path)
    return str(Path(path).resolve())


DATA_TASKS = {"write_excel": write_excel}


def _cond(c: dict, variables: dict) -> bool:
    val = _lookup(variables, c["var"], None)
    op = c.get("op", "eq")
    return {"eq": val == c.get("value"), "ne": val != c.get("value"), "exists": val not in (None, "", [])}[op]


def exception_status(error):
    return "interrupted" if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)) else "error"


# ---------------- graph execution ----------------
async def run_aop(aop: dict, overrides: dict, rt: Runtime) -> dict:
    if rt.tracing:
        rt.context.update(procedure_id=aop.get("procedure_id"), version=aop.get("version"))
    try:
        result, final_metrics = await _run_aop(aop, overrides, rt)
    except BaseException as e:
        try:
            rt.finish_run({"status": exception_status(e), "reason": f"{type(e).__name__}: {e}",
                           "error_type": type(e).__name__, "metrics": rt.metrics.snapshot()})
        except Exception:
            pass  # Finalization must not replace the original automation exception.
        raise
    rt.finish_run({**result, "metrics": final_metrics})
    return result


async def _run_aop(aop: dict, overrides: dict, rt: Runtime) -> tuple[dict, dict]:
    schema = aop.get("variables_schema", {})
    variables = {}
    for group in ("required", "optional"):
        for name, spec in schema.get(group, {}).items():
            if "default" in spec:
                variables[name] = spec["default"]
    variables.update({k: v for k, v in (overrides or {}).items() if v not in (None, "")})
    missing = [n for n in schema.get("required", {}) if variables.get(n) in (None, "")]
    if missing:
        raise ValueError(f"missing required variables: {missing}")

    g = aop["workflow_graph"]
    nodes, current = g["nodes"], g["start_node"]
    rt.emit("run_started", {"procedure_id": aop.get("procedure_id"), "version": aop.get("version"),
                            "variables": variables, "nodes": list(nodes)})
    start_url = next((render(n.get("start_url"), variables) for n in nodes.values() if n.get("start_url")), "about:blank")
    await rt.browser.start(start_url)
    result = {"status": "BLOCKED", "reason": "not started"}
    try:
        for _ in range(100):
            raw = nodes[current]
            node = render(raw, variables)
            kind = node["kind"]
            rt.emit("node_started", {"node": current, "kind": kind, "goal": node.get("goal")})
            if kind == "ui_task":
                res = await run_ui_task(node, variables, rt)
                variables.update(res.get("outputs") or {})
                rt.emit("node_finished", {"node": current, **{k: res[k] for k in ("status", "reason")}, "metrics": rt.metrics.snapshot()})
                current = {"DONE": node.get("next"), "HANDOFF": node.get("on_handoff"), "BLOCKED": node.get("on_blocked", node.get("on_handoff"))}[res["status"]]
            elif kind == "data_task":
                fn = DATA_TASKS[node["task"]]
                meta = {"procedure_id": aop.get("procedure_id"), "version": aop.get("version"), "captured_at": datetime.now().isoformat(timespec="seconds"),
                        "url": rt.browser.page.url, **{k: v for k, v in variables.items() if not isinstance(v, (list, dict))},
                        "metrics": rt.metrics.snapshot()}
                path = fn(render(node["data"], variables), node["path"], meta)
                variables[node.get("output_variable", node["task"] + "_path")] = path
                rt.emit("node_finished", {"node": current, "status": "DONE", "reason": f"wrote {path}", "metrics": rt.metrics.snapshot()})
                current = node.get("next")
            elif kind == "decision":
                current = next((r["next"] for r in node.get("rules", []) if _cond(r["when"], variables)), node.get("default_next"))
            elif kind == "end":
                result = {"status": node.get("status", "success"), "reason": node.get("message", "")}
                break
            if not current:
                result = {"status": "error", "reason": "graph has no next node"}
                break
    finally:
        final_metrics = rt.metrics.snapshot()  # retain the original pre-cleanup measurement boundary
        await rt.browser.close()
    return result, final_metrics
