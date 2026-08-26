
import os
import importlib
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:-3]
        try:
            module = importlib.import_module(f'.{module_name}', package=__name__)
            if hasattr(module, 'CustomPENGI'):
                globals()[module_name] = getattr(module, 'CustomPENGI')
            elif hasattr(module, 'ZeroShotPENGI'):
                globals()[module_name] = getattr(module, 'ZeroShotPENGI')
        except Exception as e:
            print(f"Failed to load module {module_name}: {e}")