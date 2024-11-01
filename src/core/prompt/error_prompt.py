from pydantic import ValidationError


def format_error_message(errors: list[ValidationError]) -> str:
    error_messages = [error.errors(include_url=False) for error in errors]

    final_error_messages = []
    for error in error_messages:
        error_msg = "\n".join(
            f"Field: {item['loc']}, Validation Error: {item['msg']}, but actual input is: {item['input']}"
            for item in error
        )
        final_error_messages.append(error_msg)

    formatted_message = "\n".join(str(error) for error in final_error_messages)
    return formatted_message


if __name__ == "__main__":
    from loguru import logger
    from pydantic import BaseModel

    class ErrorMessage(BaseModel):
        error_message: str

    # Create a test case with invalid data
    try:
        # Attempt to create ErrorMessage without required field
        ErrorMessage.model_validate({})
    except ValidationError as e:
        # Test the format_error_message function with the caught error
        errors = [e]
        formatted_error = format_error_message(errors=errors)

        logger.info("Formatted error message:")
        logger.info(formatted_error)

    # Test multiple validation errors
    try:
        # Attempt to create ErrorMessage with wrong type
        ErrorMessage.model_validate({"error_message": 123})  # Should be string
    except ValidationError as e:
        errors = [e]
        formatted_error = format_error_message(errors=errors)

        logger.info("\nSecond test case:")
        logger.info(formatted_error)
