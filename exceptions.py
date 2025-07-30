"""A central module for all custom exceptions used in the browser-use project."""

class AgentException(Exception):
    """Base exception for all agent-related errors for easier top-level catching."""
    pass

class AgentConfigurationError(AgentException):
    """Raised when the agent's configuration is invalid or inconsistent."""
    pass

class AgentInterruptedError(AgentException):
    """Raised when the agent's run is interrupted by an external signal (e.g., user cancellation)."""
    pass

class LLMException(AgentException):
    """Raised when there is a non-recoverable error during an LLM call."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

class RateLimitError(LLMException):
    """A specific type of LLMException raised when a rate limit is exceeded."""
    pass

class LockTimeoutError(AgentException):
    """Custom exception raised when acquiring a state lock times out, indicating a potential deadlock."""
    pass