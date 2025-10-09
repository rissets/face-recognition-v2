"""
Custom schema preprocessing hooks for drf-spectacular
"""


def custom_preprocessing_hook(endpoints):
    """
    Custom preprocessing hook to modify the schema before generation
    """
    # Add custom preprocessing logic here if needed
    return endpoints