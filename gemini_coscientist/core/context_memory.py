class ContextMemory:
    """
    Stores and retrieves the state of the AI co-scientist system.
    This is a simplified in-memory implementation.  A real implementation
    would use a database or file system.
    """
    def __init__(self):
        self.memory = {}

    def store(self, key, data):
        """Stores data in the context memory."""
        self.memory[key] = data

    def retrieve(self, key):
        """Retrieves data from the context memory."""
        return self.memory.get(key)

    def clear(self):
        """Clears the context memory."""
        self.memory = {}