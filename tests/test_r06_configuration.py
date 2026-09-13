"""New-run configuration regressions without an enclosing synthetic run scope.

The ordinary offline I/O and language-asset guards remain installed. These cases
own their fake transports and must exercise configuration before run creation.
"""

import pytest

from tests.test_effective_provider_configuration import (
    test_active_provider_snapshot_survives_mutation as test_active_provider_snapshot_survives_mutation,
)
from tests.test_effective_provider_configuration import (
    test_credentials_are_endpoint_isolated as test_credentials_are_endpoint_isolated,
)
from tests.test_effective_provider_configuration import (
    test_invalid_controls_fail_before_client_allocation as test_invalid_controls_fail_before_client_allocation,
)
from tests.test_effective_provider_configuration import (
    test_managed_run_binds_one_immutable_snapshot as test_managed_run_binds_one_immutable_snapshot,
)
from tests.test_effective_provider_configuration import (
    test_reload_uses_real_file_below_process as test_reload_uses_real_file_below_process,
)


@pytest.fixture(autouse=True)
def run_service_context() -> None:
    """Leave new-run/default tests unbound; each case owns any required services."""
