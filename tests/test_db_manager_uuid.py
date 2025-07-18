import uuid
from core.db_manager import Neo4jManagerSingleton


def test_sanitize_parameters_uuid():
    manager = Neo4jManagerSingleton()
    uid = uuid.uuid4()
    params = {"id": uid, "nested": {"inner": uid}, "list": [uid, 1]}
    sanitized = manager._sanitize_parameters(params)
    assert isinstance(sanitized["id"], str)
    assert isinstance(sanitized["nested"]["inner"], str)
    assert isinstance(sanitized["list"][0], str)
    assert sanitized["list"][1] == 1
