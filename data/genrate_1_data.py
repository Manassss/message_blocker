import pandas as pd
import random
import os

# Templates for contact-sharing messages
templates = [
    "Email me at {email}",
    "Here's my email: {email}",
    "Reach out on WhatsApp at {phone}",
    "DM me on Facebook: facebook.com/{handle}",
    "Connect on LinkedIn (linkedin.com/in/{handle})",
    "Text me at {phone}",
    "Contact: {phone}",
    "You can reach me at {phone}",
    "Snapchat me @ {handle}",
    "Message me at {email}",
    "Hit me up on WhatsApp at {phone}",

    # Obfuscated formats
    "e m a i l m e @ {email}",
    "📧 {email}",
    "c o n t a c t : {phone}",
    "t e l : {phone}",
    "my digits: {phone}",
    "find me at {email} 😉",
    "telegram: t dot me slash {handle}",
    "snap me at s-n-a-p {handle}",
    "you can DM me at {handle} on Insta",
    "hmu @ {handle}",
    "fb.com/{handle}",
    "wanna chat? {phone}",
    "my email id is {email}",
    "reach at t w o o o @example.com",
    "📱 {phone}",
    "get in touch @ {email}",
    "message me via g m a i l at {email}"
]

def generate_fake_email(index):
    return f"user{index}@example.com"

def generate_fake_phone():
    return f"+1-{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"

def generate_fake_handle():
    return f"{random.choice(['shopper', 'devguy', 'coolcat', 'fashionista'])}{random.randint(1,999)}"

def generate_restricted_dataset(n=10000):
    data = []
    for i in range(n):
        template = random.choice(templates)
        message = template.format(
            email=generate_fake_email(i),
            phone=generate_fake_phone(),
            handle=generate_fake_handle()
        )
        data.append({"message": message, "label": 1})
    
    df = pd.DataFrame(data)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/generated_restricted.csv", index=False)
    print(f"✅ Generated {n} restricted messages to data/processed/generated_restricted.csv")

if __name__ == "__main__":
    generate_restricted_dataset()