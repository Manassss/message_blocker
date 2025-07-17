import re

# Define patterns
EMAIL_PATTERN = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}"
PHONE_PATTERN = r"\b(\+?\d[\d\s\-().]{8,})\b"
SOCIAL_PATTERN = r"(?i)(instagram|facebook|snapchat|t\.me|wa\.me|linkedin|@[\w.]+|dm me|contact me|reach me)"

def contains_contact_info(message: str) -> bool:
    message_cleaned = message.replace(" ", "").lower()
    return (
        re.search(EMAIL_PATTERN, message) or
        re.search(PHONE_PATTERN, message_cleaned) or
        re.search(SOCIAL_PATTERN, message)
    )
    
if __name__ == "__main__":
    test = "D M m e a t @ ma nas_ dev"
    print("Blocked" if contains_contact_info(test) else "Clean")