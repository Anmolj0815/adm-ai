import os
from typing import Optional

def get_env_variable(key: str, default_value: Optional[str] = None) -> str:
    """
    Retrieves an environment variable and handles missing values.

    Args:
        key: The name of the environment variable.
        default_value: An optional default value if the variable is not set.

    Returns:
        The value of the environment variable.

    Raises:
        ValueError: If the environment variable is not set and no default is provided.
    """
    value = os.getenv(key, default_value)
    if value is None:
        raise ValueError(f"Environment variable '{key}' is not set.")
    return value

def format_justification(justification: str, clauses_used: list) -> str:
    """
    Formats the final justification string by including referenced clauses.
    This could be a more sophisticated function for final output formatting.
    """
    if not clauses_used:
        return justification
    
    clauses_text = "\n\nReferenced Clauses:\n" + "\n".join([f"- {clause}" for clause in clauses_used])
    return f"{justification}{clauses_text}"
