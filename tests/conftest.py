import os
import hypothesis

# Test more thoroughly during continuous integration.
hypothesis.settings.register_profile(
    "ci",
    max_examples=1000,
    print_blob=True,
    deadline=None,
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
hypothesis.settings.register_profile(
    "default",
    max_examples=100,
    deadline=None,
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)

# GitHub Actions sets the CI environment variable.
if os.getenv("CI", False):
    hypothesis.settings.load_profile("ci")
else:
    hypothesis.settings.load_profile("default")
