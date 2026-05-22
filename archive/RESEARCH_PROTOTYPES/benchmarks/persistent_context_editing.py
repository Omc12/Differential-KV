class PersistentContextEditing:
    """
    Progressive modification tests.
    Tests if changes made hours ago are correctly reflected in the current context.
    """
    def __init__(self):
        self.context_state = {}

    def apply_edit(self, key: str, value: str):
        self.context_state[key] = value

    def verify_edit(self, key: str, expected_value: str) -> bool:
        return self.context_state.get(key) == expected_value
