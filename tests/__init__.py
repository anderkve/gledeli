# workaround
# Mock importing pymultinest  -- it's not needed but may cause problems
import sys
import mock
MOCK_MODULES = ['pymultinest']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()
