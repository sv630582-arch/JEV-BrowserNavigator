"""Offline tracing contract and deterministic trace-on/off equivalence tests."""
import asyncio
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from navigator import agent, jev
from navigator.browser import StaleTarget
from navigator.runner import EventBus, Runtime, run_aop
from navigator.tracing import REDACTED, TraceWriter


def observation(number=0):
    return {"url": "https://example.test/private?token=hidden#secret", "title": "private title",
            "text": "private page text", "modal": None, "fingerprint": str(number),
            "scroll": {"y": 0, "max": 0}, "rects": [], "viewport": {"w": 100, "h": 100},
            "elements": [{"index": "1", "role": "textbox", "label": "Private name",
                          "section": "Private form", "value": "private-input" if number else "",
                          "operations": ["TYPE_TEXT"]},
                         {"index": "2", "role": "button", "label": "Next", "operations": ["CLICK"]}]}


class ScriptBrowser:
    def __init__(self, failures=(), change=True):
        self.page = SimpleNamespace(url=observation()["url"])
        self.failures = list(failures)
        self.change = change
        self.actions, self.observations, self.settles, self.starts, self.closes = [], 0, 0, 0, 0

    async def start(self, url):
        self.starts += 1

    async def close(self):
        self.closes += 1

    async def observe(self):
        self.observations += 1
        return observation(len(self.actions) if self.change else 0)

    async def settle(self):
        self.settles += 1
        return await self.observe()

    async def act(self, op, target, obs, text):
        self.actions.append((op, target, text))
        if self.failures:
            failure = self.failures.pop(0)
            if failure:
                raise failure
        return {"after": await self.settle()}

    async def active_region_html(self):
        return "<div>private page</div>"


def workflow():
    node = {"kind": "ui_task", "goal": "Enter {{name}}", "inputs": ["name"],
            "side_effects": "none", "acceptance": [{"check": "text_visible", "text": "done"}],
            "on_handoff": "handoff", "on_blocked": "handoff"}
    return {"procedure_id": "trace-fixture", "version": "1", "variables_schema": {
        "required": {"name": {"default": "private-input"}}}, "workflow_graph": {
        "start_node": "first", "nodes": {"first": {**node, "next": "second"},
        "second": {**node, "next": "done"}, "done": {"kind": "end", "status": "success"},
        "handoff": {"kind": "end", "status": "handoff"}}}}


class TracingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.quiet = redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def runtime(self, enabled=True, run_id="test-run", directory=None):
        cfg = {"tracing": {"enabled": enabled, "directory": directory or self.temp.name},
               "policy": {"max_steps": 5}}
        return Runtime(cfg, {"openrouter": "sk-private-api-key"}, EventBus(), run_id=run_id)

    def events(self, rt):
        return [json.loads(line) for line in rt.bus.trace_writer.path.read_text().splitlines()]

    def test_incremental_projection_and_redaction(self):
        rt = self.runtime()
        secret = "sk-private-api-key"
        rt.emit("run_started", {"procedure_id": "fixture", "version": "1", "variables": {"key": secret}})
        self.assertEqual(len(self.events(rt)), 1)  # readable before any terminal event
        rt.emit("node_started", {"node": "first", "kind": "ui_task", "goal": "unknown-secret"})
        with rt.step_scope(1):
            rt.emit("observation", {"obs": observation()})
            rt.emit("jev_request", {"request": {"Authorization": "Bearer " + secret}, "raw_answers": {"secret": secret}})
            rt.trace_event("action_selected", {"input_text": "unknown-secret", "input_name": "name",
                                              "selected_target": observation()["elements"][0]})
            rt.emit("error", {"message": "unknown-secret", "error_type": "ValueError", "stage": "act"})
        rt.emit("run_finished", {"status": "error", "reason": "unknown-secret"})
        events = self.events(rt)
        text = json.dumps(events)
        for forbidden in (secret, "Authorization", "unknown-secret", "private-input", "private page text", "Private name", "?token=", "/private"):
            self.assertNotIn(forbidden, text)
        self.assertEqual([e["sequence"] for e in events], list(range(1, len(events) + 1)))
        self.assertEqual(events[-1]["type"], "run_finished")
        self.assertEqual(events[2]["step_id"], "test-run:n1:s1")
        self.assertEqual(next(e for e in events if e["type"] == "error")["error_type"], "ValueError")
        self.assertEqual(next(e for e in events if e["type"] == "observation")["observation_summary"]["url"], "https:[REDACTED]")
        # Disk projection must not mutate the live UI data.
        live = next(e for e in rt.bus.history if e["type"] == "observation")
        self.assertEqual(live["obs"]["text"], "private page text")

    def test_adversarial_nested_projection_uses_only_offered_choices(self):
        rt = self.runtime()
        opaque = "Q7m9R2v8N4x6"
        rt.emit("node_started", {"node": "first", "goal": opaque})
        with rt.step_scope(1):
            body, _, _ = jev.build_request(observation(), opaque, [], {opaque: opaque}, [], None)
            rt.emit("jev_request", {"request": body, "raw_answers": {opaque: {opaque: opaque}}})
            operation = jev.parse_choice({"choice": "CLICK", "probabilities": {
                "CLICK": .9, "SCROLL_UP": .05, opaque: .05}}, {"CLICK", "WAIT"})
            target = jev.parse_choice({"choice": "2", "probabilities": {"2": .9, "999": .05, opaque: .05}}, {"2"})
            payload = {"operation": operation, "target": target,
                       "value": {"value": opaque, "dist": {opaque: .9}},
                       "flags": {"step_completed": .9, opaque: .1, "unexpected_change": {opaque: opaque}},
                       "verdict": {"verdict": "ACT", "reason": opaque, opaque: opaque},
                       "selected_target": {"index": "2", "role": "button", "label": opaque,
                                           "section": {opaque: opaque}, "source": opaque,
                                           "selected_option": {"index": opaque, "label": opaque}, opaque: opaque},
                       "observation_summary": {"url": "https://" + opaque + ".test/" + opaque,
                                               "title": opaque, "fingerprint": opaque, opaque: opaque},
                       "metrics": {"steps": 1, "input_tokens": {opaque: opaque},
                                   "time_buckets_s": {"act": .1, opaque: opaque}, opaque: opaque},
                       "results": [{"check": "text_visible", "ok": True, "detail": {opaque: opaque}, opaque: opaque}],
                       "readback": {opaque: opaque}, "error": opaque, "goal": opaque,
                       "variables": {opaque: opaque}, "outputs": {opaque: opaque},
                       "input_name": opaque, "input_text": opaque, "url_after": "https://" + opaque,
                       "error_type": opaque, "model": opaque}
            live = copy.deepcopy(payload)
            rt.emit("decision", payload)
            self.assertEqual(payload, live)
        stored = next(e for e in self.events(rt) if e["type"] == "decision")
        self.assertEqual(stored["operation"]["dist"], {"CLICK": .9})
        self.assertEqual(stored["target"]["dist"], {"2": .9})
        self.assertEqual(stored["flags"], {"step_completed": .9})
        self.assertEqual(stored["selected_target"]["index"], "2")
        self.assertEqual(stored["metrics"]["time_buckets_s"], {"act": .1})
        self.assertNotIn(opaque, rt.bus.trace_writer.path.read_text())
        # Offers cannot bleed into a later step without its own request event.
        with rt.step_scope(2):
            rt.emit("decision", {"operation": operation, "target": target})
        last = [e for e in self.events(rt) if e["type"] == "decision"][-1]
        self.assertEqual(last["operation"]["dist"], {})
        self.assertEqual(last["target"]["dist"], {})

    def test_nested_wrong_types_and_nonfinite_values_are_not_persisted(self):
        writer = TraceWriter(self.temp.name, "schema-test")
        opaque = "Q7m9R2v8N4x6"
        payload = {key: {opaque: [opaque]} for key in (
            "status", "run_id", "error_type", "latency_ms", "page_changed", "url", "flags", "metrics", "target")}
        payload.update(type="decision", operation={"value": {opaque: opaque}, "p": float("nan")},
                       selected_target={"index": {opaque: opaque}, "role": [opaque]},
                       results=[{opaque: opaque}, {"check": [opaque], "ok": True}])
        self.assertTrue(writer.write(payload))
        self.assertNotIn(opaque, writer.path.read_text())
        self.assertNotIn("NaN", writer.path.read_text())

    def test_writer_and_warning_console_failure_preserve_automation(self):
        plan = [("TYPE_TEXT", .99), ("CLICK", .99)]
        off, b0, r0, result0 = self.scenario(False, plan)
        for failure in (False, OSError("private disk failure")):
            with self.subTest(failure=failure):
                with patch.object(TraceWriter, "write", side_effect=failure if isinstance(failure, Exception) else None,
                                  return_value=False) as writer, patch("builtins.print", side_effect=BrokenPipeError("closed")):
                    on, b1, r1, result1 = self.scenario(True, plan)
                self.assertEqual((result0, b0.actions, b0.observations, r0), (result1, b1.actions, b1.observations, r1))
                self.assertEqual(writer.call_count, 1)
                self.assertEqual(sum(e["type"] == "trace_error" for e in on.bus.history), 1)
                self.assertEqual(sum(e["type"] == "run_finished" for e in on.bus.history), 1)
                self.assertTrue(on.finished)

    def test_projection_serialization_and_filesystem_failures_are_isolated(self):
        for failure in ("project", "serialize", "mkdir", "open", "flush"):
            with self.subTest(failure=failure):
                writer = TraceWriter(self.temp.name, failure)
                if failure == "project":
                    mock = patch.object(writer, "project", side_effect=ValueError("private"))
                elif failure == "serialize":
                    mock = patch.object(writer, "project", return_value={"bad": object()})
                elif failure == "mkdir":
                    mock = patch.object(Path, "mkdir", side_effect=OSError("private"))
                elif failure == "open":
                    mock = patch.object(Path, "open", side_effect=OSError("private"))
                else:
                    from unittest.mock import mock_open
                    stream = mock_open()
                    stream.return_value.flush.side_effect = OSError("private")
                    mock = patch.object(Path, "open", stream)
                with mock:
                    self.assertFalse(writer.write({"type": "run_started"}))
                self.assertTrue(writer.failed)
                self.assertFalse(writer.write({"type": "run_finished"}))

    def test_terminal_attempt_failure_fallback_and_duplicates(self):
        rt = self.runtime()
        with patch.object(rt.bus, "emit", side_effect=BrokenPipeError("closed")) as emit:
            self.assertFalse(rt.finish_run({"status": "success"}))
            self.assertEqual(emit.call_count, 2)  # bounded, no recursion
        self.assertTrue(rt.terminal_attempted)
        self.assertFalse(rt.finished)
        self.assertFalse(rt.bus.history)
        self.assertTrue(rt.finish_run({"status": "error"}))
        self.assertTrue(rt.finished)
        rt.finish_run({"status": "handoff"})
        self.assertEqual(len(self.events(rt)), 1)
        self.assertEqual(self.events(rt)[0]["status"], "success")  # preserve the original outcome

    def test_terminal_failure_then_retry_delivers_once(self):
        rt = self.runtime()
        emit = rt.bus.emit
        calls = []
        def flaky(type_, data):
            calls.append(type_)
            if len(calls) == 1:
                return False
            return emit(type_, data)
        with patch.object(rt.bus, "emit", side_effect=flaky):
            self.assertTrue(rt.finish_run({"status": "handoff"}))
        self.assertEqual(calls, ["run_finished", "run_finished"])
        self.assertEqual(len(self.events(rt)), 1)
        self.assertEqual(self.events(rt)[0]["status"], "handoff")

    def test_terminal_lost_acknowledgement_does_not_duplicate(self):
        rt = self.runtime()
        emit = rt.bus.emit
        def accepted_then_failed(type_, data):
            emit(type_, data)
            raise BrokenPipeError("ack failed")
        with patch.object(rt.bus, "emit", side_effect=accepted_then_failed) as mocked:
            self.assertTrue(rt.finish_run({"status": "success"}))
            self.assertEqual(mocked.call_count, 1)
        rt.bus.emit("run_finished", {"run_id": rt.run_id, "status": "error"})
        self.assertEqual(len(self.events(rt)), 1)

    def test_failed_terminal_delivery_does_not_repeat_automation(self):
        plan = [("TYPE_TEXT", .99), ("CLICK", .99)]
        _, b0, r0, result0 = self.scenario(False, plan)
        emit = EventBus.emit
        terminal_attempts = []
        def failing_terminal(bus, type_, data):
            if type_ == "run_finished":
                terminal_attempts.append(data)
                raise OSError("unavailable event delivery")
            return emit(bus, type_, data)
        with patch.object(EventBus, "emit", new=failing_terminal):
            rt, b1, r1, result1 = self.scenario(True, plan)
        self.assertEqual((result0, b0.actions, b0.observations, r0), (result1, b1.actions, b1.observations, r1))
        self.assertEqual(len(terminal_attempts), 2)
        self.assertFalse(rt.finished)
        self.assertTrue(rt.terminal_attempted)
        self.assertFalse(any(e["type"] == "run_finished" for e in self.events(rt)))
        self.assertTrue(rt.finish_run({"status": "error"}))
        self.assertEqual(self.events(rt)[-1]["status"], "success")

    def test_server_reports_cancellation_without_relabeling_or_swallowing(self):
        with patch("navigator.env.load_env"):
            import server
        class ImmediateThread:
            def __init__(self, target, **kwargs):
                self.target = target
            def start(self):
                self.target()
        bus = EventBus()
        failure = asyncio.CancelledError("fixture cancellation")
        cfg = {"tracing": {"enabled": True, "directory": self.temp.name}}
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "fixture-key"}), \
             patch.object(server, "CONFIG", cfg), patch.object(server, "BUS", bus), \
             patch.object(server.threading, "Thread", ImmediateThread), \
             patch.object(server, "run_aop", side_effect=failure):
            with self.assertRaises(asyncio.CancelledError) as caught:
                server.start_run({})
        self.assertIs(caught.exception, failure)
        self.assertFalse(server.RUN_LOCK.locked())
        terminals = [e for e in bus.history if e["type"] == "run_finished"]
        self.assertEqual(len(terminals), 1)
        self.assertEqual(terminals[0]["status"], "interrupted")

    def test_step_boundary_delivery_failure_does_not_escape(self):
        rt = self.runtime()
        rt.emit("node_started", {"node": "first"})
        completed = []
        with patch.object(rt, "emit", side_effect=BrokenPipeError("closed")):
            with rt.step_scope(1):
                completed.append(True)
            failure = RuntimeError("automation")
            with self.assertRaises(RuntimeError) as caught:
                with rt.step_scope(2):
                    raise failure
        self.assertEqual(completed, [True])
        self.assertIs(caught.exception, failure)
        self.assertIsNone(rt.step_id)

    def test_finalization_failure_does_not_replace_automation_exception(self):
        rt = self.runtime()
        rt.browser = ScriptBrowser()
        failure = RuntimeError("original automation failure")
        with patch.object(jev, "decide", side_effect=failure), patch.object(rt.bus, "emit", side_effect=OSError("telemetry")):
            with self.assertRaises(RuntimeError) as caught:
                asyncio.run(run_aop(workflow(), {}, rt))
        self.assertIs(caught.exception, failure)
        self.assertFalse(rt.finished)
        self.assertTrue(rt.terminal_attempted)
        self.assertEqual(rt.browser.closes, 1)

    def test_cancelled_runs_are_interrupted_and_propagate(self):
        for location in ("start", "step", "close"):
            for enabled in (False, True):
                with self.subTest(location=location, enabled=enabled):
                    rt = self.runtime(enabled, "cancel-" + location + str(enabled))
                    rt.browser = ScriptBrowser()
                    failure = asyncio.CancelledError("private cancellation")
                    aop = workflow()
                    if location == "step":
                        mock = patch.object(jev, "decide", side_effect=failure)
                    else:
                        aop["workflow_graph"]["start_node"] = "done"
                        mock = patch.object(rt.browser, location, side_effect=failure)
                    with mock, self.assertRaises(asyncio.CancelledError) as caught:
                        asyncio.run(run_aop(aop, {}, rt))
                    self.assertIs(caught.exception, failure)
                    terminals = [e for e in rt.bus.history if e["type"] == "run_finished"]
                    self.assertEqual(len(terminals), 1)
                    self.assertEqual(terminals[0]["status"], "interrupted")
                    if enabled and location == "step":
                        self.assertEqual(next(e for e in self.events(rt) if e["type"] == "step_finished")["status"], "interrupted")
                    self.assertTrue(rt.finished)

    def test_fresh_standalone_task_and_repeated_invocation(self):
        decisions, outcomes = [], []
        for enabled in (False, True):
            rt = self.runtime(enabled, "standalone-" + str(enabled))
            rt.browser = ScriptBrowser()
            with patch.object(jev, "decide", side_effect=jev.JevError("offline")) as decide:
                results = [asyncio.run(agent.run_ui_task(workflow()["workflow_graph"]["nodes"]["first"],
                                                        {"name": "private-input"}, rt)) for _ in range(2)]
            decisions.append(decide.call_args_list)
            outcomes.append((results, rt.browser.actions, rt.browser.observations))
            if enabled:
                steps = [e for e in self.events(rt) if e["type"] == "step_started"]
                self.assertEqual([e["step_id"] for e in steps], [rt.run_id + ":n1:s1", rt.run_id + ":n2:s1"])
                self.assertTrue(all(e["node"] == "standalone_ui_task" for e in steps))
        self.assertEqual(decisions[0], decisions[1])
        self.assertEqual(outcomes[0], outcomes[1])

    def test_disabled_does_not_create_trace(self):
        rt = self.runtime(False)
        rt.emit("run_started", {"procedure_id": "fixture"})
        with rt.step_scope(1):
            rt.trace_event("action_attempt_started", {"attempt": 1})
        self.assertEqual([e["type"] for e in rt.bus.history], ["run_started"])
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_writer_failure_is_reported_once_and_does_not_raise(self):
        directory = Path(self.temp.name) / "not-a-directory"
        directory.write_text("occupied")
        rt = self.runtime(directory=str(directory))
        rt.emit("run_started", {})
        rt.emit("run_finished", {"status": "success"})
        self.assertEqual(sum(e["type"] == "trace_error" for e in rt.bus.history), 1)
        self.assertEqual(rt.bus.history[-1]["status"], "success")

    def test_distinct_run_node_visits_and_partial_trace(self):
        rt = self.runtime()
        rt.emit("node_started", {"node": "revisited"})
        with rt.step_scope(1):
            pass
        rt.emit("node_started", {"node": "revisited"})
        with rt.step_scope(1):
            pass
        events = self.events(rt)
        self.assertEqual([e["step_id"] for e in events if e["type"] == "step_started"],
                         ["test-run:n1:s1", "test-run:n2:s1"])
        self.assertFalse(any(e["type"] == "run_finished" for e in events))
        other = self.runtime(run_id="other-run")
        other.emit("run_started", {})
        self.assertNotEqual(rt.bus.trace_writer.path, other.bus.trace_writer.path)

    def test_native_dialog_event_type_is_preserved(self):
        rt = self.runtime()
        dialog = SimpleNamespace(type="alert", message="private-dialog", dismiss=AsyncMock())
        asyncio.run(rt.browser._native_dialog(dialog))
        event = self.events(rt)[0]
        self.assertEqual(event["type"], "native_dialog")
        self.assertEqual(event["dialog_type"], "alert")
        self.assertEqual(event["message"], REDACTED)
        dialog.dismiss.assert_awaited_once()

    def scenario(self, enabled, plan, failures=(), accept_at=2, change=True, decision_error=None):
        rt = self.runtime(enabled, "on" if enabled else "off")
        browser = rt.browser = ScriptBrowser(failures, change)
        requests = []

        async def decide(obs, goal, rules, inputs, history, prev, key):
            body, _, _ = jev.build_request(obs, goal, rules, inputs, history, prev)
            requests.append(copy.deepcopy(body))
            if decision_error:
                raise decision_error
            op, probability = plan[min(len(requests) - 1, len(plan) - 1)]
            target = "1" if op == "TYPE_TEXT" else "2" if op == "CLICK" else None
            choice = lambda value: {"value": value, "p": .99, "dist": {value: .99}}
            return {"operation": {"value": op, "p": probability, "dist": {op: probability}},
                    "target": choice(target) if target else None,
                    "value": choice("name") if op == "TYPE_TEXT" else None,
                    "flags": {"step_completed": .99} if prev else {}, "latency_ms": 1,
                    "input_tokens": 100, "tokens_measured": True, "request": body, "raw": {"answers": {}}}

        async def check(*args):
            return len(browser.actions) >= accept_at, "private check detail"

        with patch.object(jev, "decide", side_effect=decide), patch.object(agent.checks, "run_check", side_effect=check):
            result = asyncio.run(run_aop(workflow(), {}, rt))
        return rt, browser, requests, result

    def test_trace_on_off_equivalence_and_previous_verification(self):
        plan = [("TYPE_TEXT", .99), ("CLICK", .99)]
        off, b0, requests0, result0 = self.scenario(False, plan)
        on, b1, requests1, result1 = self.scenario(True, plan)
        self.assertEqual(result0, result1)
        self.assertEqual(result1["status"], "success")
        self.assertEqual(requests0, requests1)
        self.assertEqual(b0.actions, b1.actions)
        self.assertEqual((b0.observations, b0.settles, b0.starts, b0.closes),
                         (b1.observations, b1.settles, b1.starts, b1.closes))
        self.assertEqual(len(requests1), 3)
        self.assertEqual(b1.observations, 5)
        self.assertEqual((off.metrics.steps, off.metrics.verified), (on.metrics.steps, on.metrics.verified))
        events = self.events(on)
        decisions = [e for e in events if e["type"] == "decision"]
        self.assertIsNone(decisions[0]["verifies_step_id"])
        self.assertEqual(decisions[1]["verifies_step_id"], decisions[0]["step_id"])
        self.assertIsNone(decisions[2]["verifies_step_id"])
        self.assertEqual([e["step"] for e in decisions], [1, 2, 1])
        self.assertEqual(sum(e["type"] == "step_finished" for e in events), 3)
        self.assertEqual(events[-1]["metrics"]["input_tokens"], 300)
        self.assertTrue(events[-1]["metrics"]["tokens_measured"])
        self.assertEqual(events[-1]["metrics"]["escalation_tokens"], 0)

    def test_retry_and_stale_equivalence(self):
        for failure, retry_count, iterations in ((RuntimeError("private failure"), 1, 1), (StaleTarget("gone"), 0, 2)):
            with self.subTest(failure=type(failure).__name__):
                off, b0, r0, result0 = self.scenario(False, [("TYPE_TEXT", .99)], [failure], accept_at=1)
                on, b1, r1, result1 = self.scenario(True, [("TYPE_TEXT", .99)], [failure], accept_at=1)
                self.assertEqual((result0, b0.actions, b0.observations, r0), (result1, b1.actions, b1.observations, r1))
                events = self.events(on)
                actions = [e for e in events if e["type"] == "action"]
                self.assertEqual(actions[0]["retry_count"], retry_count)
                self.assertEqual(sum(e["type"] == "action_attempt_error" for e in events), 1)
                self.assertEqual(sum(e["type"] == "step_finished" for e in events), iterations + 1)
                # Each subcase uses a fresh file on its next iteration.
                on.bus.trace_writer.path.unlink()

    def test_policy_stop_and_rejected_done_have_terminal_steps(self):
        for plan, expected_steps in (([("CLICK", .2)], 1), ([("DONE", .99)], 2)):
            with self.subTest(plan=plan):
                rt, browser, _, result = self.scenario(True, plan, accept_at=100)
                self.assertEqual(result["status"], "handoff")
                self.assertEqual(browser.actions, [])
                events = self.events(rt)
                self.assertEqual(sum(e["type"] == "step_finished" for e in events), expected_steps)
                self.assertEqual([e for e in events if e["type"] == "step_finished"][-1]["status"], "HANDOFF")
                rt.bus.trace_writer.path.unlink()

    def test_decision_failure_is_correlated(self):
        rt, browser, _, result = self.scenario(True, [], decision_error=jev.JevError("private error"))
        events = self.events(rt)
        self.assertEqual(result["status"], "handoff")
        self.assertEqual(browser.actions, [])
        self.assertEqual(next(e for e in events if e["type"] == "error")["stage"], "decide")
        self.assertEqual(next(e for e in events if e["type"] == "step_finished")["status"], "BLOCKED")

    def test_failed_readback_retries_once_but_click_does_not(self):
        for op, expected_attempts in (("TYPE_TEXT", 2), ("CLICK", 1)):
            with self.subTest(op=op):
                off, b0, r0, result0 = self.scenario(False, [(op, .99)], accept_at=100, change=False)
                on, b1, r1, result1 = self.scenario(True, [(op, .99)], accept_at=100, change=False)
                self.assertEqual((result0, b0.actions, b0.observations, r0), (result1, b1.actions, b1.observations, r1))
                events = self.events(on)
                first_step = next(e["step_id"] for e in events if e["type"] == "step_started")
                attempts = [e for e in events if e["type"] == "action_attempt_finished" and e["step_id"] == first_step]
                self.assertEqual(len(attempts), expected_attempts)
                self.assertTrue(all(not e["ok"] for e in attempts))
                self.assertTrue(all(e["execution_status"] == "returned" for e in attempts))
                on.bus.trace_writer.path.unlink()

    def test_escalation_and_repeated_direction_have_no_actions(self):
        direction = ({"diagnosis": "private diagnosis", "subgoal": "private subgoal", "candidates": [], "handoff": False},
                     {"latency_ms": 1, "usage": {"total_tokens": 2}})
        with patch.object(agent.escalate, "direction", new_callable=AsyncMock, return_value=direction) as escalate:
            off, b0, r0, result0 = self.scenario(False, [("CLICK", .45)], accept_at=100)
            calls_off = escalate.await_count
            on, b1, r1, result1 = self.scenario(True, [("CLICK", .45)], accept_at=100)
            self.assertEqual(escalate.await_count, 2 * calls_off)
        self.assertEqual((result0, b0.actions, b0.observations, r0), (result1, b1.actions, b1.observations, r1))
        self.assertEqual(b1.actions, [])
        events = self.events(on)
        self.assertEqual(sum(e["type"] == "step_finished" for e in events), 2)
        self.assertEqual(sum(e["type"] == "escalation" for e in events), 2)
        self.assertNotIn("private subgoal", json.dumps(events))

    def test_escalation_error_keeps_legacy_error_string(self):
        with patch.object(agent.escalate, "direction", side_effect=RuntimeError("private escalation error")):
            rt, browser, _, result = self.scenario(True, [("CLICK", .45)])
        self.assertEqual(result["status"], "handoff")
        self.assertEqual(browser.actions, [])
        live = next(e for e in rt.bus.history if e["type"] == "escalation")
        self.assertEqual(live["error"], "private escalation error")
        stored = next(e for e in self.events(rt) if e["type"] == "escalation")
        self.assertEqual(stored["error"], REDACTED)
        self.assertEqual(stored["error_type"], "RuntimeError")

    def test_missing_variables_finalize_without_starting_browser(self):
        rt = self.runtime()
        rt.browser = ScriptBrowser()
        aop = workflow()
        aop["variables_schema"]["required"]["name"].pop("default")
        with self.assertRaises(ValueError):
            asyncio.run(run_aop(aop, {}, rt))
        events = self.events(rt)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "run_finished")
        self.assertEqual(events[0]["procedure_id"], "trace-fixture")
        self.assertEqual(rt.browser.starts, 0)

    def test_startup_and_cleanup_errors_have_one_terminal_event(self):
        for stage in ("start", "close"):
            with self.subTest(stage=stage):
                rt = self.runtime(run_id=stage)
                rt.browser = ScriptBrowser()
                setattr(rt.browser, stage, AsyncMock(side_effect=RuntimeError("private-error")))
                aop = workflow()
                aop["workflow_graph"]["start_node"] = "done"
                with self.assertRaises(RuntimeError):
                    asyncio.run(run_aop(aop, {}, rt))
                rt.emit("run_finished", {"status": "error"})
                events = self.events(rt)
                self.assertEqual(sum(e["type"] == "run_finished" for e in events), 1)
                self.assertEqual(events[-1]["status"], "error")
                self.assertEqual(events[-1]["error_type"], "RuntimeError")

    def test_unknown_exception_finalizes_step_and_run(self):
        rt = self.runtime()
        rt.browser = ScriptBrowser()
        with patch.object(jev, "decide", side_effect=RuntimeError("private-error")):
            with self.assertRaises(RuntimeError):
                asyncio.run(run_aop(workflow(), {}, rt))
        events = self.events(rt)
        self.assertEqual(next(e for e in events if e["type"] == "step_finished")["status"], "error")
        self.assertEqual(events[-1]["type"], "run_finished")
        self.assertEqual(rt.browser.closes, 1)


if __name__ == "__main__":
    unittest.main()
