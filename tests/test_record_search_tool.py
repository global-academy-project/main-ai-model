from app.tools.record_search import make_search_records_tool


class FakeClient:
    def __init__(self):
        self.calls = []

    def call_tool(self, name, body, *, retry):
        self.calls.append((name, body, retry))
        return {"ok": True}


def test_search_records_uses_execution_scope_not_model_fields():
    client = FakeClient()
    tool = make_search_records_tool(client)
    assert tool({
        "run_id": 51,
        "site_id": 1,
        "query_type": "missing_measurements",
        "params": {"from": "2026-10-01", "to": "2026-10-31"},
    }) == {"ok": True}
    assert client.calls == [(
        "search_records",
        {
            "siteId": 1,
            "agentRunId": 51,
            "queryType": "missing_measurements",
            "params": {"from": "2026-10-01", "to": "2026-10-31"},
        },
        False,
    )]
